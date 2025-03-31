import PyPDF2
import contractions
import streamlit as st
import nltk
import glob
from pinecone import Pinecone
from typing import List, Dict, Set, Any, Callable
import collections as cl
import logging
import functools as ft
import itertools as it

import numpy as np
from numpy import ndarray as array
from pandas import DataFrame

import os
import re

from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import dotenv

dotenv.load_dotenv()

openai_api = os.getenv("OPENAI_API")
pinecone_api = os.getenv("PINECONE_API")
print(openai_api)
print(pinecone_api)
openai_client = OpenAI(api_key=openai_api)
pinecone_clinet = Pinecone(api_key=pinecone_api)
index_name = "javascript-db"
# Connecting to an Index
index = pinecone_clinet.Index(index_name)


logger: Any = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
info: Callable = logger.info
print("Hello")
logger.warning("Warning")
info = logger.warning
# logger.info("Info")
info("Info")


def var_info(a: Any) -> Any:
  """
  Args:
    a: Any  The object to print information about.
  Returns:
    Any  The first 10 rows of the object.
  Prints the type and shape of the object.
  If the object is a DataFrame, it prints the columns and data types.
  Returns the first 10 rows of the object.
  """
  type_a: Type = type(a)
  info(type_a, a.shape, sep="\n")
  if type_a == DataFrame:
    info(a.columns, a.dtypes, sep="\n")
  return a[:10]


# %% [markdown]
# # Data
# ## Loading Data
# The documents from the positive and negative folders were loaded as strings and stored in a list.  A list of labels was created indicating whether the document was from the positive folder or the negative folder.

# %%


# %% [markdown]
# ## Tokenising
# Contractions were expanded, 's was removed, and the text of each document was tokenized.

# %%


def token(string):
  return string.split(" ")


def tokenize(t: str) -> List[str]:
  """
  Tokenizes a string.
  Args:
      t: str  The string to tokenize.
  Returns:
      List[str]  A list of tokens.
  """
  t = t.lower()
  t = contractions.fix(t)
  return token(t)


def convert_pdf_to_text(pdf_path: str) -> str:
    """
    Extract plain text from a PDF using pdfminer.six.
    """
    # convert pdf to text using PyPDF2
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def save_text_to_file(text: str, output_path: str) -> None:
  """
  Save extracted text to a .txt file.
  """
  with open(output_path, "w", encoding="utf-8") as f:
    f.write(text)


def convert_pdf_to_txt_file(pdf_path: str, output_txt_path: str, start_ch, end_ch) -> None:
  print(f"📄 Extracting text from: {pdf_path}")
  text = convert_pdf_to_text(pdf_path)
  end_ch = len(text) - end_ch
  print(f"💾 Saving to: {output_txt_path}")
  save_text_to_file(text[start_ch:end_ch], output_txt_path)

  print("✅ Done!")
  return text[start_ch:end_ch]

# %% [markdown]
# ## Lemmatising
# Stopwords were removed from each list of tokens and the tokens were lemmatised.


# %%

# expand contractions
# tokens_text_pos = list(map(contractions.fix, tokens_text_pos))
# tokens_text_neg = list(map(contractions.fix, tokens_text_neg))

# not_words are stopwords and internet words
not_words: Set[str] = (set(nltk.corpus.stopwords.words('english')).difference(set(("not", "no"))))

for word in not_words:
  if word.startswith("n"):
    info(word)

# not_words: Set[str] = set(nltk.corpus.stopwords.words( 'english')).union(set(("href", "lt", "gt", "br", "p")))

info(not_words)


# apply lemmatization to the article text using NLTK
lemmatizer: Any = nltk.stem.WordNetLemmatizer()

parts_of_speech: str = "arsvn"


def lemmatize(w: str) -> str:
  """
  Lemmatizes a word as all parts of speech including nouns, verbs, adjectives, and adverbs
  Args:
      w: str  A word to be lemmatized
  Returns:
      str  The lemmatized word
  """
  for part_of_speech in parts_of_speech:
    w = lemmatizer.lemmatize(w, pos=part_of_speech)
  return w


def lemmatize_words(tokens: List[str]) -> List[str]:
  """
  Lemmatizes a list of words.
  Args:
      tokens: List[str]  A list of words to be lemmatized.
  Returns:
      List[str]  A list of lemmatized words.
  """
  return list(map(lemmatize, filter(lambda x: x not in not_words, tokens)))


def prepare_text(string):
  string = tokenize(string)
  return lemmatize_words(string)


def chunk_text(book_texts: List[str], chunk_size: int = 1000, overlap: int = 200) -> List[str]:
  splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
  chunks = []
  for text in book_texts:
    chunks.extend(splitter.split_text(text))
  return chunks

# creating embedding vector


def get_embeddings(texts: List[str]) -> List[List[float]]:
  response = openai_client.embeddings.create(
      model="text-embedding-ada-002",
      input=texts
  )
  return [e.embedding for e in response.data]


def upload_to_pinecone(texts: List[str], embeddings: List[List[float]]) -> None:
  for i, (text, embedding) in enumerate(zip(texts, embeddings)):
    index.upsert([
        (f"chunk-{i}", embedding, {"text": text})
    ])

# dfuntion to find the most similar chunk


def retrieve_similiar_chunck(query_embedding, index, top_k=1):
  response = index.query(
      vector=query_embedding,
      top_k=top_k,
      include_values=False,
      include_metadata=True
  )
  return response['matches'][0]['metadata']['text']


def prompt_builder(system_message, context):
  return system_message['content'].format(context)


system_prompt = {
    "role": "system",
    "content": """
                    we will define what system should do
                    Answer question about JavaScript using at least 50 percent of the provided context.
                    answer primarily based on the context and specify which parts are not based on context
                    Context: {}
                    """,

}


def rag_chatbot(query, openai_client):

  # Step 1: encode the query
  query_embeddings = get_embeddings(query)

  # Step 2: find the most similar chunks
  similar_chunk = retrieve_similiar_chunck(query_embeddings, index, top_k=1)

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

  return response.choices[0].message.content, augmented_prompt


st.title("RAG Chatbot with Streamlit")

# Initialize session state variables if not already present
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
      answer = rag_chatbot(user_query, openai_client=openai_client)
      st.write(answer)
  else:
    st.error("Please upload and process a PDF first.")
