import os
import io
import logging
from typing import List, Set, Any

import dotenv
import streamlit as st
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import fitz  # PyMuPDF
from bs4 import BeautifulSoup

import contractions
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone

dotenv.load_dotenv()

# API clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API"))
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API"))
index = pinecone_client.Index("javascript-db")

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NLTK setup
nltk.download('stopwords')
nltk.download('wordnet')

tokenizer = RegexpTokenizer(r"[a-z]\w*(?:'\w+)*")
lemmatizer = WordNetLemmatizer()
stop_words: Set[str] = set(stopwords.words('english')).difference({'not', 'no'})


def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = contractions.fix(text)
    return tokenizer.tokenize(text)


def lemmatize(word: str) -> str:
    for pos in ['a', 'r', 's', 'v', 'n']:
        word = lemmatizer.lemmatize(word, pos=pos)
    return word


def prepare_text(text: str) -> List[str]:
    tokens = tokenize(text)
    return [lemmatize(token) for token in tokens if token not in stop_words]


def convert_pdf_to_html(pdf_file: Any) -> str:
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    html = ""
    for page in doc:
        html += page.get_text("html")
    return html


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def chunk_text(book_texts: List[str], chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return [chunk for text in book_texts for chunk in splitter.split_text(text)]


def get_embeddings(texts: List[str]) -> List[List[float]]:
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=texts
    )
    return [e.embedding for e in response.data]


def upload_to_pinecone(texts: List[str], embeddings: List[List[float]], batch_size: int = 50) -> None:
    vectors = [(f"chunk-{i}", embedding, {"text": text})
               for i, (text, embedding) in enumerate(zip(texts, embeddings))]
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(batch)
        logger.info(f"Uploaded batch {i // batch_size + 1}/{(len(vectors) + batch_size - 1) // batch_size}")


def retrieve_similar_chunks(query_embedding: List[float], top_k: int = 3) -> str:
    response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    chunks = [match['metadata']['text'] for match in response['matches']]
    return "\n".join(chunks)


system_prompt = {
    "role": "system",
    "content": (
        "Answer questions about JavaScript using at least 50%% of the provided context. "
        "Answer primarily based on the context and clearly specify which parts are not from the context. "
        "Context: %s"
    )
}


def rag_chatbot(query: str) -> (str, str):
    query_embedding = get_embeddings([query])[0]
    context = retrieve_similar_chunks(query_embedding, top_k=3)
    augmented_prompt = system_prompt['content'] % context

    messages = [
        {"role": "system", "content": augmented_prompt},
        {"role": "user", "content": query}
    ]

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=3000
    )

    return augmented_prompt, response.choices[0].message.content


# Streamlit App
st.title("JavaScript RAG Chatbot")

if 'chunks_uploaded' not in st.session_state:
    st.session_state.chunks_uploaded = False

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

# Load default PDF if no file is uploaded and no chunks are processed
if uploaded_file is None and not st.session_state.chunks_uploaded:
    default_pdf_path = "Eloquent_JavaScript.pdf"
    if os.path.exists(default_pdf_path):
        with open(default_pdf_path, "rb") as f:
            file_bytes = f.read()
        uploaded_file = io.BytesIO(file_bytes)

if uploaded_file and not st.session_state.chunks_uploaded:
    with st.spinner("Processing PDF..."):
        html_content = convert_pdf_to_html(uploaded_file)
        raw_text = html_to_text(html_content)
        processed_text = " ".join(prepare_text(raw_text))
        chunks = chunk_text([processed_text])
        embeddings = get_embeddings(chunks)
        upload_to_pinecone(chunks, embeddings, batch_size=50)
        st.session_state.chunks_uploaded = True
        st.success("PDF processed and indexed.")

user_query = st.text_input("Ask a JavaScript question:")

if user_query:
    if st.session_state.chunks_uploaded:
        with st.spinner("Generating answer..."):
            prompt_used, answer = rag_chatbot(user_query)
            st.subheader("Prompt sent to ChatGPT:")
            st.text_area("Prompt Context:", prompt_used, height=300)
            st.subheader("ChatGPT Answer:")
            st.write(answer)
    else:
        st.warning("Please upload and process a PDF first.")
