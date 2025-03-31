import os
import logging
from typing import List, Dict, Set, Any, Callable

import dotenv
import streamlit as st
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import PyPDF2
import contractions
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone

# Load environment variables
dotenv.load_dotenv()

# Initialize API clients
openai_api_key: str = os.getenv("OPENAI_API")
pinecone_api_key: str = os.getenv("PINECONE_API")

openai_client = OpenAI(api_key=openai_api_key)
pinecone_client = Pinecone(api_key=pinecone_api_key)

# Pinecone index setup
index_name: str = "javascript-db"
index = pinecone_client.Index(index_name)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
info = logger.info

# Initialize NLTK components
nltk.download('stopwords')
nltk.download('wordnet')

reg_tokenizer = RegexpTokenizer(r"[a-z]\w*(?:'\w+)*")
lemmatizer = WordNetLemmatizer()
stop_words: Set[str] = set(stopwords.words('english')).difference({'not', 'no'})


def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = contractions.fix(text)
    return reg_tokenizer.tokenize(text)


def lemmatize(word: str) -> str:
    for pos in ['a', 'r', 's', 'v', 'n']:
        word = lemmatizer.lemmatize(word, pos=pos)
    return word


def prepare_text(text: str) -> List[str]:
    tokens = tokenize(text)
    return [lemmatize(token) for token in tokens if token not in stop_words]


def convert_pdf_to_text(pdf_file: Any) -> str:
    reader = PyPDF2.PdfReader(pdf_file)
    return "".join(page.extract_text() for page in reader.pages)


def chunk_text(book_texts: List[str], chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return [chunk for text in book_texts for chunk in splitter.split_text(text)]


def get_embeddings(texts: List[str]) -> List[List[float]]:
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=texts
    )
    return [e.embedding for e in response.data]


def upload_to_pinecone(texts: List[str], embeddings: List[List[float]]) -> None:
    vectors = [(f"chunk-{i}", embedding, {"text": text})
               for i, (text, embedding) in enumerate(zip(texts, embeddings))]
    index.upsert(vectors)


def retrieve_similar_chunk(query_embedding: List[float], top_k: int = 1) -> str:
    response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return response['matches'][0]['metadata']['text']


system_prompt = {
    "role": "system",
    "content": (
        "Answer questions about JavaScript using at least 50% of the provided context. "
        "Answer primarily based on the context and clearly specify which parts are not from the context. "
        "Context: {}"
    )
}


def rag_chatbot(query: str) -> str:
    query_embedding = get_embeddings([query])[0]
    context = retrieve_similar_chunk(query_embedding)
    augmented_prompt = system_prompt['content'].format(context)

    messages = [
        {"role": "system", "content": augmented_prompt},
        {"role": "user", "content": query}
    ]

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=3000
    )

    return response.choices[0].message.content


# Streamlit App Interface
st.title("📚 RAG JavaScript Chatbot")

if 'chunks_uploaded' not in st.session_state:
    st.session_state.chunks_uploaded = False

uploaded_file = st.file_uploader("📂 Upload PDF", type="pdf")

if uploaded_file and not st.session_state.chunks_uploaded:
    with st.spinner('🔍 Processing PDF...'):
        raw_text = convert_pdf_to_text(uploaded_file)
        processed_text = " ".join(prepare_text(raw_text))
        chunks = chunk_text([processed_text])
        embeddings = get_embeddings(chunks)
        upload_to_pinecone(chunks, embeddings)
        st.session_state.chunks_uploaded = True
        st.success("✅ PDF successfully processed and indexed!")

user_query = st.text_input("❓ Ask a JavaScript question:")

if user_query:
    if st.session_state.chunks_uploaded:
        with st.spinner('🧠 Generating answer...'):
            answer = rag_chatbot(user_query)
            st.write(answer)
    else:
        st.warning("⚠️ Please upload and process a PDF first!")
