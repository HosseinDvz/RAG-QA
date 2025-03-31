import os
import logging
from typing import List, Dict, Set, Any, Callable
import collections as cl
import itertools as it

import numpy as np
from numpy import ndarray as array
import pandas as pd
from pandas import DataFrame

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

# Define Pinecone index name
index_name: str = "javascript-db"
index = pinecone_client.Index(index_name)

# Set up logging
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
info: Callable[..., None] = logger.info


def var_info(a: Any) -> Any:
    """
    Print type and shape information about the object.
    If the object is a DataFrame, also print columns and data types.
    Returns the first 10 rows of the object.
    """
    type_a: type = type(a)
    info(type_a, getattr(a, 'shape', None), sep="\n")
    if isinstance(a, DataFrame):
        info(a.columns, a.dtypes, sep="\n")
    return a[:10]


# Initialize NLTK components
nltk.download('stopwords')
nltk.download('wordnet')

reg_tokenizer: RegexpTokenizer = RegexpTokenizer(r"[a-z]\w*(?:'\w+)*")
lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
not_words: Set[str] = set(stopwords.words('english')).difference({'not', 'no'})


def tokenize(text: str) -> List[str]:
    """
    Tokenizes a string: lowercases, expands contractions, and splits into tokens.
    """
    text = text.lower()
    text = contractions.fix(text)
    return reg_tokenizer.tokenize(text)


def lemmatize(word: str) -> str:
    """
    Lemmatizes a word across all parts of speech.
    """
    for pos in ['a', 'r', 's', 'v', 'n']:
        word = lemmatizer.lemmatize(word, pos=pos)
    return word


def lemmatize_words(tokens: List[str]) -> List[str]:
    """
    Lemmatizes a list of tokens, removing stopwords.
    """
    return [lemmatize(token) for token in tokens if token not in not_words]


def prepare_text(text: str) -> List[str]:
    """
    Tokenizes and lemmatizes input text.
    """
    tokens = tokenize(text)
    return lemmatize_words(tokens)


def convert_pdf_to_text(pdf_path: str) -> str:
    """
    Extracts text from a PDF file.
    """
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = "".join(page.extract_text() for page in pdf_reader.pages)
    return text


def save_text_to_file(text: str, output_path: str) -> None:
    """
    Saves text to a file.
    """
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(text)


def convert_pdf_to_txt_file(pdf_path: str, output_txt_path: str, start_ch: int, end_ch: int) -> str:
    """
    Converts a PDF to a text file, extracting text between specified character indices.
    """
    text = convert_pdf_to_text(pdf_path)
    extracted_text = text[start_ch:end_ch]
    save_text_to_file(extracted_text, output_txt_path)
    return extracted_text


def chunk_text(book_texts: List[str], chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Splits texts into chunks of specified size with overlap.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    for text in book_texts:
        chunks.extend(splitter.split_text(text))
    return chunks


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generates embeddings for a list of texts using OpenAI.
    """
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=texts
    )
    return [e.embedding for e in response.data]


def upload_to_pinecone(texts: List[str], embeddings: List[List[float]]) -> None:
    """
    Uploads text chunks and their embeddings to Pinecone.
    """
    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        index.upsert([
            (f"chunk-{i}", embedding, {"text": text})
        ])


def retrieve_similar_chunk(query_embedding: List[float], top_k: int = 1) -> str:
    """
    Retrieves the most similar text chunk from Pinecone based on the query embedding.
    """
    response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_values=False,
        include_metadata=True
    )
    return response['matches'][0]['metadata']['text']


def prompt_builder(system_message: Dict[str, str], context: str) -> str:
    """
    Builds a prompt by formatting the system message with the provided context.
    """
    return system_message['content'].format(context)


system_prompt: Dict[str, str] = {
    "role": "system",
    "content": (
        "Answer questions about JavaScript using at least 50 percent of the provided context. "
        "Answer primarily based on the context and specify which parts are not based on context. "
        "Context: {}"
    ),
}


def rag_chatbot(query: str) -> str:
    """
    Processes a user query using a Retrieval-Augmented Generation approach.
    """
    query_embedding = get_embeddings([query])[0]
    similar_chunk = retrieve_similar_chunk(query_embedding)
    augmented_prompt = prompt_builder(system_prompt, similar_chunk)

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


# Streamlit Application
st.title("RAG Chatbot with Streamlit")

# Initialize session state variables
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False

if 'chunks_uploaded' not in st.session_state:
    st.session_state.chunks_uploaded = False

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file and not st.session_state.pdf_processed:
    with st.spinner('Processing PDF...'):
        output_txt = "javascript_book_1.txt"
        text = convert_pdf_to_txt_file(uploaded_file, output_txt, 50*500, 10*500+41220)
        text_tokens = prepare_text(text)
        text = " ".join(text_tokens)
        chunks = chunk_text([text])
        embeddings = get_embeddings(chunks)
        upload_to_pinecone(chunks, embeddings=embeddings)
        st.session_state.pdf_processed = True
        st.session_state.chunks_uploaded = True

    st.success("PDF processed and indexed!")

user_query = st.text_input("Ask your question:")

if user_query:
    if st.session_state.chunks_uploaded:
        with st.spinner('Generating answer...'):
            answer = rag_chatbot(user_query)
            st.write(answer)
    else:
        st.error("Please upload and process a PDF first.")