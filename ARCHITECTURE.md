# Technical Architecture

## Components
- **Streamlit UI:** upload controls, chat interface, status metrics, and source display.
- **Document loader:** extracts text from PDF pages and TXT files, retaining source/page metadata.
- **Chunker:** creates overlapping character-based chunks to preserve context across boundaries.
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` maps chunks and questions into vectors.
- **FAISS:** stores normalized vectors and performs inner-product similarity search (equivalent to cosine similarity for normalized vectors).
- **Answer generation:** optional Groq chat completion receives retrieved passages and a grounded-answer instruction. Without a key, retrieved evidence is shown directly.

## Data flow
### Ingestion
Upload -> Extract -> Chunk -> Embed -> Add vectors to FAISS

### Query
Question -> Embed -> Top-k search -> Context assembly -> LLM/fallback -> Answer + sources

## Design choices
- A compact sentence embedding model keeps semantic retrieval practical on a typical development laptop.
- Overlap helps reduce loss of meaning at chunk boundaries.
- Source metadata is attached to every chunk so users can inspect where retrieved evidence originated.
- The system explicitly instructs the LLM not to invent information outside the retrieved context.
