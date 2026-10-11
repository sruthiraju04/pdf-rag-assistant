import gc
import os
import shutil
import time
import uuid
from pathlib import Path

import streamlit as st

import config
import file_utils
import rag_pipeline


st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=":books:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    html, body, [class*="css"], .stMarkdown {
        font-family: 'Outfit', sans-serif;
    }
    .stDeployButton,
    [data-testid="stDeployButton"],
    [data-testid="stAppDeployButton"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"] {
        display: none !important;
    }

    .main-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6C5CE7, #a29bfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #636e72;
        font-size: 1.15rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    .stAlert {
        border-radius: 16px !important;
        border: 1px solid rgba(108, 92, 231, 0.1) !important;
    }
    .sidebar-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #2d3436;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }
    .stButton>button {
        border-radius: 10px;
        border: 1px solid rgba(108, 92, 231, 0.2);
        transition: all 0.3s ease;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #6C5CE7 !important;
        color: white !important;
        border-color: #6C5CE7 !important;
        box-shadow: 0 4px 12px rgba(108, 92, 231, 0.2);
    }
</style>
""",
    unsafe_allow_html=True,
)


def safe_rmtree(path: Path) -> None:
    """Remove a directory tree, ignoring locked-file errors from Chroma/Windows."""
    if path.exists():
        try:
            shutil.rmtree(path)
        except Exception:
            pass


def cleanup_old_sessions() -> None:
    """Delete temporary files and session vector stores older than 24 hours."""
    cutoff = time.time() - 24 * 3600

    if config.DB_DIR.exists():
        for path in config.DB_DIR.iterdir():
            if path.is_dir() and path.name.startswith("session_"):
                try:
                    if path.stat().st_mtime < cutoff:
                        shutil.rmtree(path)
                except Exception:
                    pass

    if config.TEMP_DIR.exists():
        for path in config.TEMP_DIR.iterdir():
            try:
                if path.stat().st_mtime >= cutoff:
                    continue
                if path.is_file():
                    os.remove(path)
                elif path.is_dir():
                    shutil.rmtree(path)
            except Exception:
                pass


def build_document_ids(file_hash: str, documents) -> list[str]:
    return [f"{file_hash}-{idx}" for idx, _ in enumerate(documents)]


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "processed_file_ids" not in st.session_state:
    st.session_state.processed_file_ids = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "api_key" not in st.session_state:
    st.session_state.api_key = config.OPENROUTER_API_KEY or ""
if "cleanup_done" not in st.session_state:
    cleanup_old_sessions()
    st.session_state.cleanup_done = True

session_db_dir = config.DB_DIR / f"session_{st.session_state.session_id}"
session_temp_dir = config.TEMP_DIR / f"session_{st.session_state.session_id}"

st.markdown('<div class="main-title">PDF RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload PDFs, retrieve grounded context, and chat with answers using LangChain + OpenRouter.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="sidebar-header">Configuration</div>', unsafe_allow_html=True)

    if config.OPENROUTER_API_KEY:
        st.success("API key set via environment (.env)")
    else:
        api_key_input = st.text_input(
            "OpenRouter API Key",
            value=st.session_state.api_key,
            type="password",
            help="Get an API key from openrouter.ai.",
        )
        if api_key_input:
            st.session_state.api_key = api_key_input

    retrieval_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=8,
        value=config.DEFAULT_RETRIEVAL_K,
        help="Number of chunks retrieved for each question.",
    )

    st.divider()

    st.markdown('<div class="sidebar-header">Document Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a PDF document",
        type=["pdf"],
        key=f"pdf_uploader_{st.session_state.uploader_key}",
        help=f"PDF only. Max size: {config.MAX_UPLOAD_SIZE_MB} MB.",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash = file_utils.sha256_bytes(file_bytes)
        safe_name = file_utils.sanitize_filename(uploaded_file.name)

        if file_hash not in st.session_state.processed_file_ids:
            if not config.is_configured(st.session_state.api_key):
                st.error("Please configure your OpenRouter API key first.")
            else:
                progress_container = st.container()
                with progress_container:
                    status_text = st.empty()
                    progress_bar = st.progress(0)
                    temp_path = session_temp_dir / f"{file_hash[:12]}_{safe_name}"

                    try:
                        file_utils.validate_pdf_upload(file_bytes, config.MAX_UPLOAD_SIZE_MB)

                        session_temp_dir.mkdir(parents=True, exist_ok=True)
                        status_text.text("Saving file locally...")
                        progress_bar.progress(25)

                        with open(temp_path, "wb") as f:
                            f.write(file_bytes)

                        status_text.text("Loading and splitting PDF...")
                        progress_bar.progress(50)
                        chunks = rag_pipeline.load_and_split_pdf(str(temp_path))

                        for chunk_index, chunk in enumerate(chunks):
                            chunk.metadata["file_hash"] = file_hash
                            chunk.metadata["chunk_index"] = chunk_index

                        status_text.text(f"Generating embeddings for {len(chunks)} chunks...")
                        progress_bar.progress(75)
                        document_ids = build_document_ids(file_hash, chunks)

                        if st.session_state.vector_store is not None:
                            st.session_state.vector_store.add_documents(
                                chunks,
                                ids=document_ids,
                            )
                        else:
                            db = rag_pipeline.create_vector_store(
                                chunks,
                                st.session_state.api_key,
                                persist_dir=str(session_db_dir),
                                ids=document_ids,
                            )
                            st.session_state.vector_store = db

                        st.session_state.processed_files.append(safe_name)
                        st.session_state.processed_file_ids.append(file_hash)

                        progress_bar.progress(100)
                        status_text.text("Ingestion completed successfully.")
                        st.success(f"Indexed: {safe_name}")
                    except file_utils.UploadValidationError as exc:
                        progress_bar.empty()
                        status_text.empty()
                        st.error(str(exc))
                    except Exception as exc:
                        progress_bar.empty()
                        status_text.empty()
                        st.error(f"Failed to process PDF: {exc}")
                    finally:
                        if temp_path.exists():
                            try:
                                os.remove(temp_path)
                            except Exception:
                                pass
                        try:
                            if session_temp_dir.exists() and not os.listdir(session_temp_dir):
                                os.rmdir(session_temp_dir)
                        except Exception:
                            pass
        else:
            st.info(f"{safe_name} is already indexed in this session.")

    if st.session_state.vector_store is not None:
        st.divider()
        if st.button("Clear Database & History", use_container_width=True):
            with st.spinner("Resetting database..."):
                old_session_db_dir = session_db_dir
                st.session_state.vector_store = None
                st.session_state.processed_files = []
                st.session_state.processed_file_ids = []
                st.session_state.chat_history = []
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.uploader_key += 1
                gc.collect()
                safe_rmtree(old_session_db_dir)
                st.success("Database and history reset successfully.")
                st.rerun()

    st.divider()

    with st.expander("Connection Info"):
        st.markdown(f"**Chat Model:** `{config.OPENROUTER_CHAT_MODEL}`")
        st.markdown(f"**Embedding Model:** `{config.OPENROUTER_EMBEDDING_MODEL}`")
        st.markdown(f"**Session ID:** `{st.session_state.session_id[:8]}...`")

if not config.is_configured(st.session_state.api_key):
    st.warning("Please enter your OpenRouter API key in the sidebar to start.")
elif st.session_state.vector_store is None:
    st.info("Please upload a PDF document in the sidebar to index it.")
else:
    st.success(f"Currently Querying: {', '.join(st.session_state.processed_files)}")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Ask a question about the document..."):
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching document context and generating answer..."):
                try:
                    response = rag_pipeline.query_rag_pipeline(
                        st.session_state.vector_store,
                        user_query,
                        st.session_state.api_key,
                        search_kwargs={"k": retrieval_k},
                    )

                    answer = response["answer"]

                    st.markdown(answer)

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )
                except Exception as exc:
                    st.error(f"An error occurred: {exc}")
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": f"Failed to retrieve an answer due to an error: {exc}",
                        }
                    )
