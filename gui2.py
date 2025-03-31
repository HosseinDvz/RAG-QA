import streamlit as st
import PyPDF2
import contractions
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from typing import List, Set, Any, Callable, Tuple, Dict
import collections as cl
import itertools as it
import logging
import os
import dotenv
from pinecone import Pinecone
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load environment variables
dotenv.load_dotenv()

openai_api = os.getenv("OPENAI_API")
pinecone_api = os.getenv("PINECONE_API")
openai_client = OpenAI(api_key=openai_api)
pinecone_client = Pinecone(api_key=pinecone_api)
index_name = "javascript-db"
index = pinecone_client.Index(index_name)

# Logger setup
logger: Any = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
info: Callable = logger.info

# Tokenization setup
reg_tokenizer = RegexpTokenizer(r"[a-z]\w*(?:'\w+)*")
lemmatizer = WordNetLemmatizer()
not_words: Set[str] = set(nltk.corpus.stopwords.words('english')).difference({"not", "no"})

# Functions for text processing


def token(string: str) -> List[str]:
    return reg_tokenizer.tokenize(string.lower())


def tokenize(text: str) -> List[str]:
    text = contractions.fix(text.lower())
    return token(text)


def lemmatize(w: str) -> str:
    for pos in "arsvn":
        w = lemmatizer.lemmatize(w, pos=pos)
    return w


def lemmatize_words(tokens: List[str]) -> List[str]:
    return [lemmatize(word) for word in tokens if word not in not_words]


def prepare_text(text: str) -> List[str]:
    return lemmatize_words(tokenize(text))


def convert_pdf_to_text(pdf_path: str) -> str:
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    return "".join(page.extract_text() for page in pdf_reader.pages)


def chunk_text(texts: List[str], chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    return chunks


def get_embeddings(texts: List[str]) -> List[List[float]]:
    response = openai_client.embeddings.create(model="text-embedding-ada-002", input=texts)
    return [e.embedding for e in response.data]


def upload_to_pinecone(texts: List[str], embeddings: List[List[float]]) -> None:
    index.upsert([(f"chunk-{i}", emb, {"text": txt}) for i, (txt, emb) in enumerate(zip(texts, embeddings))])


def retrieve_similar_chunk(query_embedding, top_k=1) -> str:
    response = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return response['matches'][0]['metadata']['text']


def prompt_builder(context: str) -> str:
    return f"""Answer question about JavaScript using at least 50% context.
Context: {context}"""


def rag_chatbot(query: str) -> str:
    query_embedding = get_embeddings([query])[0]
    similar_chunk = retrieve_similar_chunk(query_embedding)
    prompt = prompt_builder(similar_chunk)

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": query}
    ]

    response = openai_client.chat.completions.create(model="gpt-4o", messages=messages, max_tokens=3000)
    return response.choices[0].message.content


# Streamlit App
st.title("Enhanced RAG Chatbot")

if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file and not st.session_state.pdf_processed:
    with st.spinner('Processing PDF...'):
        text = convert_pdf_to_text(uploaded_file)
        prepared_text = " ".join(prepare_text(text))
        chunks = chunk_text([prepared_text])
        embeddings = get_embeddings(chunks)
        upload_to_pinecone(chunks, embeddings)
        st.session_state.pdf_processed = True
    st.success("PDF processed and indexed!")

user_query = st.text_input("Ask your question:")

if user_query:
    if st.session_state.pdf_processed:
        with st.spinner('Generating answer...'):
            answer = rag_chatbot(user_query)
            st.write(answer)
    else:
        st.error("Please upload and process a PDF first.")
