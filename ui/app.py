# ui/enh_app.py
import streamlit as st
import os
import sys
import time
from typing import List
from streamlit.runtime.uploaded_file_manager import UploadedFile

# -----------------------------------------------------------------------------
# Project-root imports
# -----------------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
from agents.coordinator_agent import CoordinatorAgent   # noqa: E402

# -----------------------------------------------------------------------------
# Streamlit page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="🤖 Agentic RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom CSS
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header       {text-align:center;color:#1f77b4;padding:1rem 0;}
    .success-message   {background:#d4edda;border:1px solid #c3e6cb;
                        color:#155724;padding:0.75rem;border-radius:6px;}
    .error-message     {background:#f8d7da;border:1px solid #f5c6cb;
                        color:#721c24;padding:0.75rem;border-radius:6px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
DOCS_DIR = os.path.join(ROOT_DIR, "documents")

# -----------------------------------------------------------------------------
# Helper – Save uploaded files
# -----------------------------------------------------------------------------
def save_uploaded_files(files: List[UploadedFile]) -> List[str]:
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)

    paths: list[str] = []
    for f in files:
        if f.size > 50 * 1024 * 1024:
            st.error(f"❌ {f.name} is larger than 50 MB – skipped")
            continue
        path = os.path.join(DOCS_DIR, f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        paths.append(path)
        print(f"Saved: {f.name}")
    return paths

# -----------------------------------------------------------------------------
# Initialise Coordinator + Agents
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def init_system() -> CoordinatorAgent:
    with st.spinner("🚀 Initialising agents …"):
        coord = CoordinatorAgent()
        coord.start_all_agents()
        return coord

coordinator = init_system()

# -----------------------------------------------------------------------------
# Session-state initialisation
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []          # chat history
if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = True        # True ⇒ can query

# -----------------------------------------------------------------------------
# Helper – Poll ingestion status → update kb_ready
# -----------------------------------------------------------------------------
def poll_ingestion():
    if getattr(coordinator, "ingestion_status", None):
        if not st.session_state.kb_ready:
            st.session_state.kb_ready = True

poll_ingestion()

# -----------------------------------------------------------------------------
# ---------------------------  SIDEBAR  ---------------------------------------
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("📚 Knowledge-Base Manager")

    # System status
    if st.session_state.kb_ready:
        st.success("✅ KB ready – you may query")
    else:
        st.info("⏳ Ingesting documents … please wait")

    st.markdown("---")

    # File uploader
    uploaded = st.file_uploader(
        "Upload documents",
        accept_multiple_files=True,
        type=["pdf", "docx", "pptx", "csv", "txt", "md"],
    )

    # Process button
    if st.button("🔄 Process documents", disabled=not st.session_state.kb_ready):
        if uploaded:
            paths = save_uploaded_files(uploaded)
            if paths:
                st.session_state.kb_ready = False
                coordinator.handle_document_ingestion(paths)
                st.success(
                    f"📨 {len(paths)} file(s) sent for ingestion – watch terminal logs"
                )
                time.sleep(1)
                st.rerun()
        else:
            st.warning("Upload at least one file before processing")

    # Clear KB
    if st.button("🗑️ Clear knowledge base", disabled=not st.session_state.kb_ready):
        coordinator.handle_clear_knowledge_base()
        st.session_state.kb_ready = False
        st.success("KB cleared – please re-ingest documents")
        time.sleep(1)
        st.rerun()

    st.markdown("---")
    st.subheader("🤖 Agent status")
    st.markdown(
        "🎯 **Coordinator** – running\n\n"
        "📥 **Ingestion** – idle\n\n"
        "🔍 **Retrieval** – idle\n\n"
        "🧠 **LLM** – idle"
    )

# -----------------------------------------------------------------------------
# ---------------------------   MAIN PAGE  ------------------------------------
# -----------------------------------------------------------------------------
st.markdown('<h1 class="main-header">🤖 Agentic RAG Chatbot</h1>',
            unsafe_allow_html=True)
st.write(
    "Ask questions **after** documents are ingested. "
    "Source passages are displayed under each answer."
)

# ---------------- Chat history display --------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📖 Sources", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    st.info(f"**{i}.** {src}")

# ---------------- Chat input -------------------------------------------------
prompt = st.chat_input(
    "Ask me anything about your documents …",
    disabled=not st.session_state.kb_ready,
)

if prompt:
    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response placeholder
    with st.chat_message("assistant"):
        ph = st.empty()
        with st.spinner("🧠 Thinking …"):
            resp = coordinator.handle_query(prompt)
        ph.markdown(resp.answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": resp.answer, "sources": resp.source_chunks}
        )

        if resp.source_chunks and not any(s.startswith("Error:") for s in resp.source_chunks):
            with st.expander("📖 Sources", expanded=False):
                for i, src in enumerate(resp.source_chunks, 1):
                    st.info(f"**{i}.** {src}")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "🏗️ **Architecture** : Multi-Agent (Coordinator / Ingestion / Retrieval / LLM)   |  "
    "🤖 **LLM** : Gemini 2.5 Pro   |  "
    "📊 **Vector DB** : FAISS   |  "
    "🔄 **Message Broker** : Redis Pub/Sub"
)
        