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
```

## 🔑 Setup Instructions

- Clone the repository:
```
git clone https://github.com/HosseinDvz/RAG-QA
cd RAG-QA
```
 - Create a .env file in the root directory:
```
OPENAI_API=your-openai-api-key
PINECONE_API=your-pinecone-api-key
```

 - Run the Streamlit app:
```
streamlit run AskMyDoc.py
```
## 💸 Important Notes
### Pinecone:
Pinecone offers a generous free plan — enough for small to medium datasets.

### OpenAI Embeddings (text-embedding-ada-002):
⚠️ NOT free — you are billed based on the amount of text embedded.

Current cost is roughly $0.0001 per 1,000 tokens (as of 2024).

Short documents are cheap; large PDFs can cost more. Always monitor your usage!

### ChatGPT 4o:
You also pay per completion depending on model/token limits (small cost for answers).

## 🧠 How It Works
- Upload a PDF file

- Extract document text

- Extract title (from metadata or GPT if missing)

- Clean and chunk the document

- Generate vector embeddings (OpenAI API)

- Store embeddings in Pinecone

- Ask a question

- Retrieve top similar chunks (with score threshold 0.8)

- Build a system prompt with retrieved context

- Generate final answer from ChatGPT

- Display context + answer in the UI

## 🔮 Future Updates
We are committed to making AskMyDoc more accessible and cost-effective. To achieve this, we plan to:

Explore Alternative Embedding Methods: Investigate open-source or more affordable embedding models to replace the current paid OpenAI embeddings.​

Integrate Gemini for Response Generation: Utilize Google's Gemini API for generating answers, leveraging its generous free tier and advanced capabilities.​
Google AI Studio

By implementing these changes, we aim to reduce or eliminate usage costs, making AskMyDoc freely available to a broader audience.
