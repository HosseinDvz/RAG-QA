import os
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from openai import OpenAI
from pinecone import Pinecone
import dotenv
from typing import List, Any, Tuple
dotenv.load_dotenv()

# API clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API"))
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API"))
index = pinecone_client.Index("javascript-db")

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_file: Any) -> Tuple[str, str]:
    """Extract raw text and book title from a PDF file. Use GPT to infer title if metadata is missing."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    
    full_text = ""
    for page in doc:
        full_text += page.get_text("text")

    metadata = doc.metadata
    title = metadata.get('title', '').strip()

    # Fallback: use ChatGPT to guess title from first page
    if not title:
        first_page_text = doc[0].get_text("text").strip()
        title = guess_title_with_gpt(first_page_text)

    return full_text, title or "Unknown Title"


def guess_title_with_gpt(first_page_text: str) -> str:
    """Use ChatGPT to guess the book/document title from first page text."""
    prompt = (
        "You are an AI trained to extract document titles. "
        "Given the first page of a document, identify the most likely title. "
        "Respond only with the title, nothing else.\n\n"
        f"First Page Text:\n{first_page_text[:1500]}"  # truncate to keep it focused
    )

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt}
        ],
        max_tokens=20,
        temperature=0.3
    )

    title = response.choices[0].message.content.strip()
    return title


def clean_text(text: str) -> str:
    """Basic cleaning: remove excessive whitespace."""
    return " ".join(text.split())


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Chunk clean text using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)


def get_embeddings(texts: List[str], model: str = "text-embedding-ada-002") -> List[List[float]]:
    """Get embeddings from OpenAI for a list of text chunks."""
    if not texts:
        return []

    response = openai_client.embeddings.create(
        model=model,
        input=texts
    )
    embeddings = [item.embedding for item in response.data]
    return embeddings


def upload_to_pinecone(texts: List[str], embeddings: List[List[float]], batch_size: int = 50) -> None:
    """Upload embeddings and texts to Pinecone in batches."""
    if not texts or not embeddings:
        logger.warning("No texts or embeddings provided for upload.")
        return

    vectors = [
        (f"chunk-{i}", embedding, {"text": text})
        for i, (text, embedding) in enumerate(zip(texts, embeddings))
    ]

    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        logger.info(f"Uploaded batch {i // batch_size + 1}/{(len(vectors) + batch_size - 1) // batch_size}")



def process_pdf_and_upload_to_pinecone(pdf_file: Any) -> str:
    """Full pipeline: Extract text, chunk, embed, upload to Pinecone, and return book title."""
    raw_text, title = extract_text_from_pdf(pdf_file)
    cleaned_text = clean_text(raw_text)
    chunks = chunk_text(cleaned_text)
    embeddings = get_embeddings(chunks)
    upload_to_pinecone(chunks, embeddings)
    logger.info(f"Processing complete: {len(chunks)} chunks uploaded.")

    return title



def retrieve_similar_chunks(query_embedding: List[float], top_k: int = 3, score_threshold: float = 0.8) -> str:
    """Retrieve similar chunks from Pinecone with a minimum similarity threshold."""
    response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    matches = response.get('matches', [])
    if not matches:
        return ""

    # Filter based on score threshold
    filtered_chunks = [
        match['metadata']['text']
        for match in matches
        if match.get('score', 0) >= score_threshold and 'metadata' in match and 'text' in match['metadata']
    ]

    if not filtered_chunks:
        return "", "No relevant context found. Please refine your query.", ""

    return "\n".join(filtered_chunks)


system_prompt_template = (
    "Answer questions about uploaded book using at least 70% of the provided context. "
    "Answer primarily based on the context and must specify which parts are not from the context.\n\n"
    "Context:\n{}"
)

def rag_chatbot(query: str) -> Tuple[str, str, str]:
    """Run a full RAG pipeline: Embed query, retrieve context, prompt LLM, return answer."""
    query_embedding = get_embeddings([query])[0]
    context = retrieve_similar_chunks(query_embedding, top_k=3)

    if not context:
        return "", "No relevant context found. Please refine your query.", ""

    augmented_prompt = system_prompt_template.format(context)

    messages = [
        {"role": "system", "content": augmented_prompt},
        {"role": "user", "content": query}
    ]

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=3000
    )

    return augmented_prompt, response.choices[0].message.content, context


import streamlit as st

st.set_page_config(layout="wide")
st.title("AskMyDoc (RAG)")

if 'chunks_uploaded' not in st.session_state:
    st.session_state.chunks_uploaded = False
if 'book_title' not in st.session_state:
    st.session_state.book_title = None  # Initialize book title safely

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file and not st.session_state.chunks_uploaded:
    with st.spinner("Processing PDF..."):
        # Process uploaded PDF with the new pipeline
        title = process_pdf_and_upload_to_pinecone(uploaded_file)
        st.session_state.book_title = title  # Save the title in session
        st.session_state.chunks_uploaded = True
        st.success("PDF processed, chunked, embedded, and uploaded to Pinecone.")

elif not uploaded_file:
    st.info("Please upload a PDF file to start.")

# Show the book title **after upload**
if st.session_state.book_title:
    st.subheader(f"Book Title: {st.session_state.book_title}")

user_query = st.text_input("Ask a question from this book:")

if user_query:
    if st.session_state.chunks_uploaded:
        with st.spinner("Generating answer..."):
            prompt_used, answer, context = rag_chatbot(user_query)
            st.subheader("Prompt sent to ChatGPT:")
            st.text_area("Prompt Context:", prompt_used, height=300)
            st.subheader("ChatGPT Answer:")
            st.write(answer)
    else:
        st.warning("Please upload and process a PDF first.")

