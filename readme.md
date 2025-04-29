# AskMyDoc 📚🤖

> Upload a PDF, ask questions, and get AI-powered answers based on the document's real content.

---

## 🚀 Overview

**AskMyDoc** is a smart Retrieval-Augmented Generation (RAG) chatbot application.  
It allows users to upload a **PDF book** or **document**, automatically extract and chunk its content, generate semantic embeddings, and **ask any question** based on the uploaded material.

The system uses:
- **OpenAI's `text-embedding-ada-002`** model for high-quality vector embeddings (⚡ Paid API)
- **Pinecone** for scalable vector database storage (✅ Free plan available)
- **ChatGPT (gpt-4o)** for final natural language answers
- **Streamlit** for a clean, interactive frontend

---

## 🛠 Features

- 📄 **Upload any PDF** (book, guide, manual, etc.)
- 🧠 **Automatic title extraction** (using PDF metadata or GPT-4 inference from the first page)
- 🔥 **Chunk large documents** smartly for efficient retrieval
- 🔎 **Semantic search** for relevant context with high similarity filtering (≥80%)
- 📝 **Context-aware answering** powered by GPT-4o
- 🎯 **Displays retrieved context and final generated answer**
- 🌐 **Runs on Streamlit** — ready for fast deployment or sharing
- 📦 **Modular architecture** — easy to extend or adapt to other document types

---

## 📋 Requirements

- Python 3.8+
- OpenAI API key (for embeddings + GPT-4o completions)
- Pinecone API key (for vector database)
- Streamlit
- LangChain (for text chunking)
- PyMuPDF (for PDF reading)

### Install dependencies:
```bash
pip install -r requirements.txt

