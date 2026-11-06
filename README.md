# 🧠 KNOWLEDGE AI — RAG-Based AI Knowledge Assistant

An intelligent document-based AI assistant that allows users to upload PDF and TXT files, retrieve relevant information, and receive context-aware answers using Retrieval-Augmented Generation (RAG).

## 🚀 Overview

KNOWLEDGE AI transforms documents into an interactive knowledge base.

Instead of manually searching through lengthy documents, users can ask questions in natural language and receive relevant answers grounded in their uploaded content.

## ✨ Key Features

* 📄 PDF and TXT document support
* ✂️ Intelligent text chunking
* 🧠 Semantic embeddings using Sentence Transformers
* 🔍 Similarity-based document retrieval
* ⚡ FAISS vector search
* 🤖 AI-powered response generation using Groq
* 💾 Persistent FAISS knowledge base
* 📚 Source-aware responses
* 🔐 Environment-based API key configuration
* 🌐 Interactive Streamlit interface

## 🛠️ Technology Stack

| Technology            | Purpose                   |
| --------------------- | ------------------------- |
| Python                | Core programming          |
| Streamlit             | User interface            |
| Sentence Transformers | Text embeddings           |
| FAISS                 | Vector similarity search  |
| NumPy                 | Numerical operations      |
| Groq API              | AI response generation    |
| PyPDF                 | PDF text extraction       |
| Python-dotenv         | Environment configuration |

## 🏗️ System Architecture

1. **Document Upload:** Users upload PDF or TXT files.
2. **Text Extraction:** The application extracts document content.
3. **Chunking:** Large text is divided into smaller overlapping chunks.
4. **Embedding Generation:** Sentence Transformers converts chunks into numerical vectors.
5. **Vector Indexing:** FAISS stores the embeddings for efficient similarity search.
6. **Question Processing:** User questions are converted into embeddings.
7. **Retrieval:** FAISS identifies the most relevant document chunks.
8. **Answer Generation:** Groq generates answers using the retrieved context.
9. **Persistent Storage:** The FAISS index and document metadata are saved locally for reuse.

## 📂 Project Structure

```text
KNOWLEDGE_AI_RAG/
│
├── app.py
│
├── app/
│   ├── document_loader.py
│   ├── chunker.py
│   └── rag_pipeline.py
│
├── data/
│   └── faiss_store/
│       ├── knowledge_base.faiss
│       └── documents.json
│
├── sample_documents/
│   ├── AI_Introduction.txt
│   └── RAG_Architecture.txt
│
├── screenshots/
│
├── .env.example
├── .gitignore
├── requirements.txt
├── ARCHITECTURE.md
├── DEMO_SCRIPT.md
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate into the project

```bash
cd KNOWLEDGE_AI_RAG
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure environment variables

Create a `.env` file in the project root.

Add:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

Replace the placeholder with your own API key.

**Important:** Never upload your `.env` file or expose your API key on GitHub.

### 7. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🧪 Sample Questions

* What is Artificial Intelligence?
* Explain RAG architecture step by step.
* What are embeddings?
* How does FAISS perform similarity search?
* Explain the document retrieval process.

## 💾 Persistent Knowledge Base

KNOWLEDGE AI uses FAISS for vector indexing and JSON for document storage.

The application saves:

* `knowledge_base.faiss` — FAISS vector index.
* `documents.json` — Document chunks and metadata.

Previously indexed documents can be restored when the application is restarted, provided the saved files remain available.

## 🎯 Project Objectives

* Simplify document-based information retrieval.
* Demonstrate practical implementation of Retrieval-Augmented Generation.
* Combine semantic search with generative AI.
* Reduce manual searching through lengthy documents.
* Build an interactive AI-powered knowledge assistant.

## 🔮 Future Enhancements

* Multiple document collections.
* Chat history and conversation memory.
* Advanced document citation.
* Support for additional file formats.
* Cloud-based vector database integration.
* User authentication and document management.

## 👩‍💻 Project Information

**Project:** KNOWLEDGE AI — RAG-Based AI Knowledge Assistant

**Domain:** Artificial Intelligence and Data Science

**Category:** Generative AI · NLP · Information Retrieval

**Interface:** Streamlit

---

*Built to make document understanding smarter, faster, and more accessible.*
