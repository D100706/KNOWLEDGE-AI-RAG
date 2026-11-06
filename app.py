import streamlit as st
from pathlib import Path
import tempfile

from app.document_loader import load_document
from app.rag_pipeline import RAGPipeline


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="KNOWLEDGE AI | RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0b1020, #111827);
        color: #f8fafc;
    }

    h1, h2, h3 {
        color: #f8fafc !important;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #60a5fa;
        margin-bottom: 0;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .metric-card {
        background: #172033;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #26344d;
        text-align: center;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 800;
        color: #60a5fa;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 14px;
    }

    .stButton > button {
        background: #2563eb;
        color: white;
        border: none;
        border-radius: 9px;
        padding: 10px 18px;
        font-weight: 600;
    }

    .stButton > button:hover {
        background: #1d4ed8;
        color: white;
    }

    [data-testid="stSidebar"] {
        background: #0f172a;
    }

    .footer {
        text-align: center;
        color: #64748b;
        padding: 25px;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "rag" not in st.session_state:
    st.session_state.rag = RAGPipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_names" not in st.session_state:
    st.session_state.indexed_names = []


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🧠 KNOWLEDGE AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Your intelligent document-based RAG assistant</div>',
    unsafe_allow_html=True
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("## 📚 Knowledge Base")

    st.write(
        "Upload PDF or TXT documents to build your personal AI knowledge base."
    )

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if st.button("⚡ Process Documents", use_container_width=True):

        if not uploaded_files:
            st.warning("Please upload at least one document.")

        else:

            all_docs = []
            failed_files = []

            progress = st.progress(0)
            status = st.empty()

            for index, uploaded_file in enumerate(uploaded_files):

                status.info(f"Processing {uploaded_file.name}...")

                try:
                    suffix = Path(uploaded_file.name).suffix

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=suffix
                    ) as temp_file:

                        temp_file.write(uploaded_file.getvalue())
                        temp_path = temp_file.name

                    try:
                        docs = load_document(temp_path)

                        if docs:
                            for doc in docs:
                                if isinstance(doc, dict):
                                    doc["metadata"] = doc.get("metadata", {})
                                    doc["metadata"]["source"] = uploaded_file.name
                                else:
                                    doc.metadata["source"] = uploaded_file.name

                            all_docs.extend(docs)

                    finally:
                        Path(temp_path).unlink(missing_ok=True)

                except Exception as error:
                    failed_files.append(
                        f"{uploaded_file.name}: {error}"
                    )

                progress.progress((index + 1) / len(uploaded_files))

            if all_docs:

                try:
                    added_chunks = st.session_state.rag.add_documents(all_docs)

                    st.session_state.indexed_names = sorted(
                        set(
                            doc["metadata"]["source"]
                            if isinstance(doc, dict)
                            else doc.metadata.get("source", "Unknown")
                            for doc in st.session_state.rag.documents
                        )
                    )

                    st.session_state.messages = []

                    status.success(
                        f"Successfully indexed {added_chunks} text chunks!"
                    )

                    st.rerun()

                except Exception as error:
                    st.error(f"Indexing failed: {error}")

            else:
                status.error("No readable text was found in the uploaded documents.")

            for failure in failed_files:
                st.error(failure)

    st.divider()

    st.markdown("### 📄 Indexed Documents")

    if st.session_state.indexed_names:

        for name in st.session_state.indexed_names:
            st.markdown(f"📄 {name}")

    else:
        st.info("No documents indexed yet.")

    st.divider()

    if st.button("🗑️ Clear Knowledge Base", use_container_width=True):

        st.session_state.rag = RAGPipeline()
        st.session_state.messages = []
        st.session_state.indexed_names = []

        st.success("Knowledge base cleared.")
        st.rerun()

    st.caption("KNOWLEDGE AI | RAG System")


# --------------------------------------------------
# DASHBOARD METRICS
# --------------------------------------------------

stats = st.session_state.rag.get_statistics()

documents_count = stats.get("documents", 0)
chunks_count = stats.get("chunks", 0)
index_status = stats.get("index_status", "Inactive")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{documents_count}</div>
            <div class="metric-label">Documents Indexed</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{chunks_count}</div>
            <div class="metric-label">Text Chunks</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{index_status}</div>
            <div class="metric-label">Retrieval Engine</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.divider()


# --------------------------------------------------
# CHAT INTERFACE
# --------------------------------------------------

st.markdown("## 💬 Chat with Your Documents")

st.write(
    "Ask questions based on the information contained in your uploaded documents."
)

# Display previous chat messages

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("📚 View Retrieved Sources"):

                for source in message["sources"]:
                    st.markdown(f"**Source:** {source}")


# Chat input

question = st.chat_input(
    "Ask something about your uploaded documents..."
)


if question:

    if not st.session_state.rag.documents:

        st.warning(
            "Please upload and process documents before asking questions."
        )

    else:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Searching your knowledge base..."):

                try:

                    result = st.session_state.rag.query(question)

                    if isinstance(result, dict):

                        answer = result.get(
                            "answer",
                            "I could not generate an answer."
                        )

                        sources = result.get("sources", [])

                    else:

                        answer = str(result)
                        sources = []

                    st.markdown(answer)

                    if sources:

                        with st.expander("📚 View Retrieved Sources"):

                            for source in sources:
                                st.markdown(f"**Source:** {source}")

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        }
                    )

                except Exception as error:

                    error_message = f"Something went wrong: {error}"

                    st.error(error_message)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_message
                        }
                    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        KNOWLEDGE AI — Retrieval-Augmented Generation Assistant
        <br>
        Built with Python, Streamlit, Sentence Transformers and FAISS
    </div>
    """,
    unsafe_allow_html=True
)

