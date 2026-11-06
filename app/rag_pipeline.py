import json
import os
from pathlib import Path

import numpy as np
import faiss

from sentence_transformers import SentenceTransformer


class RAGPipeline:

    def __init__(
        self,
        model_name="all-MiniLM-L6-v2",
        chunk_size=800,
        chunk_overlap=150,
        top_k=4,
        storage_dir="data/faiss_store"
    ):

        # Load environment variables
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        # Persistent storage directory
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = (
            self.storage_dir / "knowledge_base.faiss"
        )

        self.documents_path = (
            self.storage_dir / "documents.json"
        )

        # Embedding model
        self.embedding_model = SentenceTransformer(model_name)

        # Knowledge base
        self.documents = []
        self.index = None

        self.embedding_dimension = (
            self.embedding_model.get_sentence_embedding_dimension()
        )

        # Groq integration
        self.groq_client = None

        self.groq_model = os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-120b"
        )

        self._initialize_groq()

        # Automatically restore saved knowledge base
        self.load_knowledge_base()

    # --------------------------------------------------
    # GROQ INITIALIZATION
    # --------------------------------------------------

    def _initialize_groq(self):

        try:
            from groq import Groq

            api_key = os.getenv("GROQ_API_KEY")

            if api_key:
                self.groq_client = Groq(api_key=api_key)

        except Exception:
            self.groq_client = None

    # --------------------------------------------------
    # TEXT CHUNKING
    # --------------------------------------------------

    def chunk_text(self, text):

        if not text or not text.strip():
            return []

        words = text.split()

        chunks = []

        start = 0

        while start < len(words):

            end = min(
                start + self.chunk_size,
                len(words)
            )

            chunk = " ".join(
                words[start:end]
            ).strip()

            if chunk:
                chunks.append(chunk)

            if end >= len(words):
                break

            start = end - self.chunk_overlap

        return chunks

    # --------------------------------------------------
    # DOCUMENT NORMALIZATION
    # --------------------------------------------------

    def _normalize_document(self, document):

        if isinstance(document, dict):

            text = document.get("text", "")
            metadata = document.get("metadata", {})

        else:

            text = getattr(
                document,
                "page_content",
                ""
            )

            metadata = getattr(
                document,
                "metadata",
                {}
            )

        if not isinstance(metadata, dict):
            metadata = {}

        return str(text).strip(), metadata.copy()

    # --------------------------------------------------
    # ADD DOCUMENTS
    # --------------------------------------------------

    def add_documents(self, documents):

        if not documents:
            return 0

        added_chunks = 0

        for document in documents:

            text, metadata = self._normalize_document(
                document
            )

            if not text:
                continue

            chunks = self.chunk_text(text)

            for chunk_number, chunk in enumerate(
                chunks,
                start=1
            ):

                chunk_record = {
                    "text": chunk,
                    "metadata": {
                        **metadata,
                        "chunk": chunk_number
                    }
                }

                self.documents.append(chunk_record)

                added_chunks += 1

        if added_chunks > 0:

            self._rebuild_index()

            # Save updated knowledge base automatically
            self.save_knowledge_base()

        return added_chunks

    # --------------------------------------------------
    # EMBEDDING GENERATION
    # --------------------------------------------------

    def generate_embeddings(self, texts):

        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return np.asarray(
            embeddings,
            dtype=np.float32
        )

    # --------------------------------------------------
    # FAISS INDEX
    # --------------------------------------------------

    def _rebuild_index(self):

        if not self.documents:

            self.index = None
            return

        texts = [
            document["text"]
            for document in self.documents
        ]

        embeddings = self.generate_embeddings(texts)

        self.index = faiss.IndexFlatIP(
            self.embedding_dimension
        )

        self.index.add(embeddings)

    # --------------------------------------------------
    # SAVE KNOWLEDGE BASE
    # --------------------------------------------------

    def save_knowledge_base(self):

        self.storage_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        if not self.documents or self.index is None:

            self._delete_saved_files()
            return False

        temporary_index = (
            self.storage_dir / "knowledge_base.tmp.faiss"
        )

        temporary_documents = (
            self.storage_dir / "documents.tmp.json"
        )

        try:

            # Save FAISS index
            faiss.write_index(
                self.index,
                str(temporary_index)
            )

            # Save document text and metadata
            with temporary_documents.open(
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    self.documents,
                    file,
                    ensure_ascii=False,
                    indent=2,
                    default=str
                )

            # Replace old files with new files
            os.replace(
                temporary_index,
                self.index_path
            )

            os.replace(
                temporary_documents,
                self.documents_path
            )

            return True

        finally:

            # Remove temporary files if any remain
            if temporary_index.exists():
                temporary_index.unlink()

            if temporary_documents.exists():
                temporary_documents.unlink()

    # --------------------------------------------------
    # LOAD SAVED KNOWLEDGE BASE
    # --------------------------------------------------

    def load_knowledge_base(self):

        if (
            not self.index_path.exists()
            or not self.documents_path.exists()
        ):
            return False

        try:

            # Load document information
            with self.documents_path.open(
                "r",
                encoding="utf-8"
            ) as file:

                saved_documents = json.load(file)

            # Load FAISS index
            saved_index = faiss.read_index(
                str(self.index_path)
            )

            if not isinstance(saved_documents, list):

                raise ValueError(
                    "Invalid documents file format."
                )

            if saved_index.ntotal != len(saved_documents):

                raise ValueError(
                    "FAISS index and documents do not match."
                )

            if saved_index.d != self.embedding_dimension:

                raise ValueError(
                    "Embedding dimensions do not match."
                )

            self.documents = saved_documents
            self.index = saved_index

            print(
                f"Restored {len(self.documents)} document chunks."
            )

            return True

        except Exception as error:

            self.documents = []
            self.index = None

            print(
                f"Could not load saved knowledge base: {error}"
            )

            return False

    # --------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------

    def retrieve(self, question, top_k=None):

        if not self.documents or self.index is None:
            return []

        if not question or not question.strip():
            return []

        limit = top_k or self.top_k

        query_embedding = self.generate_embeddings(
            [question]
        )

        scores, indices = self.index.search(
            query_embedding,
            min(limit, len(self.documents))
        )

        results = []

        for score, index_position in zip(
            scores[0],
            indices[0]
        ):

            if index_position < 0:
                continue

            document = self.documents[index_position]

            results.append({
                "text": document["text"],
                "metadata": document["metadata"],
                "score": float(score)
            })

        return results

    # --------------------------------------------------
    # GROQ ANSWER GENERATION
    # --------------------------------------------------

    def generate_answer(
        self,
        question,
        retrieved_documents
    ):

        if not retrieved_documents:

            return (
                "I couldn't find relevant information "
                "in your uploaded documents."
            )

        context = "\n\n".join(
            [
                f"Source: {doc['metadata'].get('source', 'Unknown')}\n"
                f"Content: {doc['text']}"
                for doc in retrieved_documents
            ]
        )

        # Fallback if Groq is not configured
        if self.groq_client is None:

            return (
                "AI answer generation is not configured yet. "
                "Here are the most relevant passages retrieved "
                "from your documents:\n\n"
                + context
            )

        system_prompt = """
You are KNOWLEDGE AI, an intelligent document-based assistant.

Rules:
1. Answer using the supplied document context.
2. Do not invent facts that are not present in the context.
3. If the answer is unavailable, clearly say so.
4. Give clear, structured and useful explanations.
5. Use bullet points when appropriate.
6. Maintain technical accuracy.
"""

        user_prompt = f"""
DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

Provide an accurate answer based on the document context.
"""

        try:

            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                temperature=0.2,
                max_tokens=1200
            )

            return response.choices[0].message.content

        except Exception as error:

            return (
                "The AI generation service encountered an error. "
                "Retrieved document passages are shown below.\n\n"
                + context
                + f"\n\nTechnical details: {error}"
            )

    # --------------------------------------------------
    # COMPLETE RAG QUERY
    # --------------------------------------------------

    def query(self, question):

        retrieved_documents = self.retrieve(question)

        answer = self.generate_answer(
            question,
            retrieved_documents
        )

        sources = []

        for document in retrieved_documents:

            metadata = document["metadata"]

            source_name = metadata.get(
                "source",
                "Unknown"
            )

            page = metadata.get("page")

            source_label = (
                f"{source_name} — Page {page}"
                if page
                else source_name
            )

            if source_label not in sources:
                sources.append(source_label)

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_documents": retrieved_documents
        }

    # --------------------------------------------------
    # KNOWLEDGE BASE STATISTICS
    # --------------------------------------------------

    def get_statistics(self):

        unique_documents = set()

        for document in self.documents:

            source = document["metadata"].get(
                "source",
                "Unknown"
            )

            unique_documents.add(source)

        return {
            "documents": len(unique_documents),

            "chunks": len(self.documents),

            "index_status": (
                "FAISS Active"
                if self.index is not None
                else "Inactive"
            ),

            "storage_status": (
                "Saved to disk"
                if (
                    self.index_path.exists()
                    and self.documents_path.exists()
                )
                else "Not saved"
            )
        }

    # --------------------------------------------------
    # DELETE SAVED KNOWLEDGE BASE
    # --------------------------------------------------

    def _delete_saved_files(self):

        for path in (
            self.index_path,
            self.documents_path
        ):

            if path.exists():
                path.unlink()

    # --------------------------------------------------
    # CLEAR KNOWLEDGE BASE
    # --------------------------------------------------

    def clear(self):

        self.documents = []
        self.index = None

        # Delete persistent files too
        self._delete_saved_files()

        return True