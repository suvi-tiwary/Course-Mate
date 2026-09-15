import os
import shutil
import tempfile
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


# =========================================================
# LOAD ENVIRONMENT
# =========================================================

load_dotenv()


# =========================================================
# API KEY
# =========================================================

def get_api_key():
    """Read API key from Streamlit Cloud secrets or local environment."""

    try:
        secret_key = (
            st.secrets.get("GEMINI_API_KEY")
            or st.secrets.get("GOOGLE_API_KEY")
        )
    except FileNotFoundError:
        secret_key = None

    return (
        secret_key
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )


GEMINI_API_KEY = get_api_key()


if not GEMINI_API_KEY:
    st.error(
        "🔑 API key is missing. Add GEMINI_API_KEY to Streamlit Cloud Secrets "
        "or to your local .env file, then restart the app."
    )
    st.stop()


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Course-Mate",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# PREMIUM DARK UI
# =========================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont,
        'Segoe UI', sans-serif;
    }

    /* =========================================
       GLOBAL
       ========================================= */

    .stApp {
        background: linear-gradient(
            135deg,
            #0a0810 0%,
            #0f0d1a 50%,
            #08070d 100%
        ) !important;

        color: #F5F3FF !important;
    }

    .main {
        background: transparent !important;
    }

    .block-container {
        max-width: 1300px;
        padding-top: 2rem !important;
        padding-bottom: 8rem !important;
    }


    /* =========================================
       HEADER
       ========================================= */

    header[data-testid="stHeader"] {
        background: rgba(10, 8, 16, 0.5) !important;
        backdrop-filter: blur(15px) !important;
        border-bottom: 1px solid rgba(124, 58, 237, 0.1) !important;
    }

    header[data-testid="stHeader"] > div {
        background: transparent !important;
    }

    header[data-testid="stHeader"] button {
        color: #9B8FB8 !important;
        transition: all 0.3s ease !important;
    }

    header[data-testid="stHeader"] button:hover {
        color: #7C3AED !important;
    }


    /* =========================================
       SIDEBAR
       ========================================= */

    section[data-testid="stSidebar"] {
        background: rgba(12, 10, 20, 0.7) !important;
        backdrop-filter: blur(10px) !important;
        border-right: 1px solid rgba(124, 58, 237, 0.1) !important;
    }

    section[data-testid="stSidebar"] > div {
        background: transparent !important;
    }

    section[data-testid="stSidebar"] * {
        color: #D8D1E3;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #F8F7FF !important;
        font-weight: 700 !important;
    }


    /* =========================================
       TEXT
       ========================================= */

    h1,
    h2,
    h3 {
        color: #F8F7FF !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }

    h1 {
        font-size: 2.5rem !important;
    }

    h2 {
        font-size: 1.75rem !important;
    }

    h3 {
        font-size: 1.25rem !important;
    }

    p {
        color: #AFA5BE !important;
        line-height: 1.7 !important;
    }

    label {
        color: #BEB4CC !important;
        font-weight: 600 !important;
    }


    /* =========================================
       BUTTONS
       ========================================= */

    .stButton > button {
        background: linear-gradient(
            135deg,
            #7C3AED 0%,
            #6D28D9 100%
        ) !important;

        color: #FFFFFF !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        padding: 0.75rem 1.5rem !important;

        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;

        box-shadow:
            0 4px 15px rgba(124, 58, 237, 0.2) !important;
    }

    .stButton > button:hover {
        background: linear-gradient(
            135deg,
            #8B5CF6 0%,
            #7C3AED 100%
        ) !important;

        border-color: #A78BFA !important;

        box-shadow:
            0 8px 30px rgba(124, 58, 237, 0.35) !important;

        transform: translateY(-2px) !important;
    }

    .stButton > button:active {
        transform: translateY(0px) !important;
    }


    /* =========================================
       FILE UPLOADER
       ========================================= */

    [data-testid="stFileUploader"] {
        background: rgba(124, 58, 237, 0.05) !important;
        border: 2px dashed rgba(124, 58, 237, 0.3) !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        transition: all 0.3s ease !important;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: rgba(124, 58, 237, 0.6) !important;
        background: rgba(124, 58, 237, 0.08) !important;
    }

    [data-testid="stFileUploader"] section {
        background: transparent !important;
    }

    [data-testid="stFileUploader"] small {
        color: #81768F !important;
    }

    [data-testid="stFileUploaderFile"] {
        background: rgba(124, 58, 237, 0.1) !important;
        border: 1px solid rgba(124, 58, 237, 0.2) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }


    /* =========================================
       CHAT MESSAGES
       ========================================= */

    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
    }

    [data-testid="stChatMessage"] p {
        color: #D8D1E3 !important;
        line-height: 1.8 !important;
    }

    [data-testid="stChatMessage"] li {
        color: #D8D1E3 !important;
        line-height: 1.8 !important;
    }

    [data-testid="stChatMessage"] strong {
        color: #F5F3FF !important;
        font-weight: 700 !important;
    }

    [data-testid="stChatMessage"] code {
        background: rgba(124, 58, 237, 0.15) !important;
        color: #C4B5FD !important;
        padding: 0.3rem 0.6rem !important;
        border-radius: 6px !important;
        font-size: 0.9em !important;
    }

    [data-testid="stChatMessage"]:has(
        [data-testid="chatAvatarIcon-user"]
    ) {
        background: rgba(124, 58, 237, 0.1) !important;
        border: 1px solid rgba(124, 58, 237, 0.2) !important;
        border-radius: 16px !important;
        padding: 1.2rem !important;
        margin-bottom: 1rem !important;
    }

    [data-testid="stChatMessage"]:has(
        [data-testid="chatAvatarIcon-assistant"]
    ) {
        background: transparent !important;
        padding: 1rem 0 !important;
        margin-bottom: 2rem !important;
    }


    /* =========================================
       CHAT INPUT
       ========================================= */

    [data-testid="stChatInput"] {
        background: rgba(15, 12, 22, 0.8) !important;
        backdrop-filter: blur(10px) !important;
        border: 1.5px solid rgba(124, 58, 237, 0.3) !important;
        border-radius: 16px !important;

        box-shadow:
            0 8px 32px rgba(124, 58, 237, 0.1) !important;

        transition: all 0.3s ease !important;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: #7C3AED !important;

        box-shadow:
            0 0 0 2px rgba(124, 58, 237, 0.2),
            0 12px 40px rgba(124, 58, 237, 0.15) !important;
    }

    [data-testid="stChatInput"] > div {
        background: transparent !important;
        border: none !important;
    }

    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: #F5F3FF !important;
        caret-color: #A78BFA !important;
        border: none !important;
        font-size: 1rem !important;
        line-height: 1.5 !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #766B84 !important;
    }

    [data-testid="stChatInput"] button {
        background: linear-gradient(
            135deg,
            #7C3AED 0%,
            #6D28D9 100%
        ) !important;

        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        transition: all 0.2s ease !important;
        font-weight: 700 !important;
    }

    [data-testid="stChatInput"] button:hover {
        background: linear-gradient(
            135deg,
            #8B5CF6 0%,
            #7C3AED 100%
        ) !important;

        box-shadow:
            0 6px 20px rgba(124, 58, 237, 0.3) !important;

        transform: translateY(-2px) !important;
    }


    /* =========================================
       TEXT INPUT
       ========================================= */

    div[data-baseweb="input"] {
        background: rgba(16, 13, 23, 0.6) !important;
        border-color: rgba(124, 58, 237, 0.2) !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="input"] input {
        background: transparent !important;
        color: #F5F3FF !important;
        font-size: 1rem !important;
    }

    div[data-baseweb="input"] input::placeholder {
        color: #766B84 !important;
    }


    /* =========================================
       CONTAINERS
       ========================================= */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(16, 13, 23, 0.5) !important;
        border: 1px solid rgba(124, 58, 237, 0.15) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        transition: all 0.3s ease !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(124, 58, 237, 0.3) !important;
        background: rgba(16, 13, 23, 0.7) !important;
    }


    /* =========================================
       ALERTS
       ========================================= */

    div[data-testid="stAlert"] {
        border-radius: 12px !important;
        border: 1px solid !important;
        padding: 1rem !important;
        background-color: rgba(0, 0, 0, 0.2) !important;
    }


    /* =========================================
       EXPANDERS
       ========================================= */

    details {
        border: 1px solid rgba(124, 58, 237, 0.15) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        background: rgba(16, 13, 23, 0.5) !important;
        transition: all 0.3s ease !important;
    }

    details:hover {
        border-color: rgba(124, 58, 237, 0.3) !important;
        background: rgba(16, 13, 23, 0.7) !important;
    }

    summary {
        color: #F5F3FF !important;
        font-weight: 700 !important;
        cursor: pointer !important;
        transition: color 0.2s ease !important;
    }

    summary:hover {
        color: #A78BFA !important;
    }


    /* =========================================
       DIVIDER
       ========================================= */

    hr {
        border-color: rgba(124, 58, 237, 0.15) !important;
        margin: 1.5rem 0 !important;
    }


    /* =========================================
       SCROLLBAR
       ========================================= */

    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: transparent;
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(124, 58, 237, 0.2);
        border-radius: 10px;
        transition: background 0.2s ease;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: rgba(124, 58, 237, 0.4);
    }


    /* =========================================
       FEATURE CARDS
       ========================================= */

    .feature-card {
        padding: 24px;

        background: linear-gradient(
            135deg,
            rgba(124, 58, 237, 0.1) 0%,
            rgba(139, 92, 246, 0.05) 100%
        );

        border: 1px solid rgba(124, 58, 237, 0.2);
        border-radius: 16px;
        min-height: 180px;

        display: flex;
        flex-direction: column;
        justify-content: center;

        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .feature-card:hover {
        border-color: rgba(124, 58, 237, 0.4);

        background: linear-gradient(
            135deg,
            rgba(124, 58, 237, 0.15) 0%,
            rgba(139, 92, 246, 0.08) 100%
        );

        box-shadow:
            0 8px 24px rgba(124, 58, 237, 0.15);

        transform: translateY(-4px);
    }

    .feature-icon {
        font-size: 36px;
        margin-bottom: 12px;
    }

    .feature-title {
        color: #F5F3FF;
        font-weight: 700;
        font-size: 16px;
        margin-bottom: 8px;
    }

    .feature-desc {
        color: #8E849B;
        font-size: 13px;
        line-height: 1.6;
    }


    /* =========================================
       STATUS CARD
       ========================================= */

    .status-card {
        padding: 16px;

        background: linear-gradient(
            135deg,
            rgba(16, 65, 48, 0.2) 0%,
            rgba(22, 68, 59, 0.1) 100%
        );

        border: 1px solid rgba(134, 239, 172, 0.3);
        border-radius: 12px;
        transition: all 0.3s ease;
    }

    .status-card:hover {
        border-color: rgba(134, 239, 172, 0.5);

        background: linear-gradient(
            135deg,
            rgba(16, 65, 48, 0.3) 0%,
            rgba(22, 68, 59, 0.15) 100%
        );
    }

    .status-title {
        color: #86EFAC;
        font-weight: 700;
        font-size: 14px;
        margin-bottom: 4px;
    }

    .status-desc {
        color: #6EE7B7;
        font-size: 12px;
    }


    /* =========================================
       INFO CARD
       ========================================= */

    .info-card {
        padding: 16px;
        background: rgba(16, 13, 23, 0.5);
        border: 1px solid rgba(124, 58, 237, 0.1);
        border-radius: 12px;
        transition: all 0.3s ease;
    }

    .info-card:hover {
        border-color: rgba(124, 58, 237, 0.2);
        background: rgba(16, 13, 23, 0.7);
    }

    .info-text {
        color: #8E849B;
        font-size: 13px;
        line-height: 1.6;
    }


    /* =========================================
       ANIMATIONS
       ========================================= */

    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }

        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(20px);
        }

        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    [data-testid="stChatMessage"] {
        animation: fadeIn 0.4s ease-out;
    }


    /* =========================================
       MOBILE
       ========================================= */

    @media (max-width: 768px) {

        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 1rem !important;
        }

        h1 {
            font-size: 1.8rem !important;
        }

        h2 {
            font-size: 1.4rem !important;
        }

        .feature-card {
            min-height: 150px;
            padding: 16px;
        }

        [data-testid="stChatInput"] {
            border-radius: 12px !important;
        }

        .stButton > button {
            padding: 0.6rem 1rem !important;
        }
    }


    /* =========================================
       REMOVE STREAMLIT BRANDING
       ========================================= */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# PATHS
# =========================================================

# IMPORTANT:
# Do NOT use "./chroma_db" directly.
# Streamlit Cloud can run the app from a different working directory.
# We therefore use a guaranteed writable temporary directory.

BASE_TEMP_DIR = Path(tempfile.gettempdir()) / "course_mate"

BASE_TEMP_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# SESSION ID
# =========================================================

if "session_id" not in st.session_state:

    st.session_state.session_id = uuid.uuid4().hex


# =========================================================
# SESSION-SPECIFIC DIRECTORIES
# =========================================================

SESSION_DIR = (
    BASE_TEMP_DIR /
    st.session_state.session_id
)

CHROMA_DIR = SESSION_DIR / "chroma_db"
TEMP_DIR = SESSION_DIR / "temp"


CHROMA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TEMP_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


if "document_name" not in st.session_state:
    st.session_state.document_name = None


if "document_pages" not in st.session_state:
    st.session_state.document_pages = 0


if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = 0


if "processed" not in st.session_state:
    st.session_state.processed = False


# =========================================================
# EMBEDDINGS
# =========================================================

@st.cache_resource
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# =========================================================
# LLM
# =========================================================

@st.cache_resource
def get_llm():

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=GEMINI_API_KEY,
    )


# =========================================================
# PROCESS PDF
# =========================================================

def process_pdf(uploaded_file):
    """Process uploaded PDF and create vector store."""

    # -----------------------------------------
    # Make sure directories exist
    # -----------------------------------------

    TEMP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # -----------------------------------------
    # Save uploaded PDF
    # -----------------------------------------

    # Avoid problematic filename paths
    safe_filename = Path(uploaded_file.name).name

    file_path = TEMP_DIR / safe_filename


    with open(file_path, "wb") as f:

        f.write(
            uploaded_file.getbuffer()
        )


    # -----------------------------------------
    # Load PDF
    # -----------------------------------------

    loader = PyPDFLoader(
        str(file_path)
    )

    documents = loader.load()


    if not documents:

        raise ValueError(
            "The PDF contains no readable pages."
        )


    # -----------------------------------------
    # Split documents
    # -----------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(
        documents
    )


    if not chunks:

        raise ValueError(
            "Could not create chunks from the PDF."
        )


    # -----------------------------------------
    # Create embeddings
    # -----------------------------------------

    embeddings = get_embeddings()


    # -----------------------------------------
    # Remove old Chroma database
    # -----------------------------------------

    if CHROMA_DIR.exists():

        shutil.rmtree(
            CHROMA_DIR,
            ignore_errors=True
        )


    # -----------------------------------------
    # Recreate Chroma directory
    # -----------------------------------------

    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # -----------------------------------------
    # Create Chroma vector store
    # -----------------------------------------

    vectorstore = Chroma.from_documents(

        documents=chunks,

        embedding=embeddings,

        persist_directory=str(
            CHROMA_DIR
        )
    )


    # -----------------------------------------
    # Save in session state
    # -----------------------------------------

    st.session_state.vectorstore = vectorstore

    st.session_state.document_name = (
        uploaded_file.name
    )

    st.session_state.document_pages = (
        len(documents)
    )

    st.session_state.document_chunks = (
        len(chunks)
    )

    st.session_state.processed = True

    st.session_state.messages = []


    return (
        len(documents),
        len(chunks)
    )


# =========================================================
# ASK COURSE-MATE
# =========================================================

def ask_course_mate(question):
    """Process user question and return answer with sources."""

    vectorstore = (
        st.session_state.vectorstore
    )


    if vectorstore is None:

        return (
            "Please upload and process your course PDF first.",
            []
        )


    # -----------------------------------------
    # Retriever
    # -----------------------------------------

    retriever = vectorstore.as_retriever(

        search_type="mmr",

        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )


    # -----------------------------------------
    # Retrieve documents
    # -----------------------------------------

    docs = retriever.invoke(
        question
    )


    if not docs:

        return (
            "I couldn't find relevant information "
            "in your notes. Try rephrasing your question.",
            []
        )


    # -----------------------------------------
    # Build context
    # -----------------------------------------

    context_parts = []


    for doc in docs:

        page = (
            doc.metadata.get("page", 0)
            + 1
        )

        context_parts.append(
            f"[Page {page}]\n"
            f"{doc.page_content}"
        )


    context = "\n\n---\n\n".join(
        context_parts
    )


    # -----------------------------------------
    # Prompt
    # -----------------------------------------

    prompt = f"""
You are Course-Mate, an AI study assistant.

Answer the student's question using ONLY the provided course material.

Guidelines:
- If the answer is not in the material, say so clearly.
- Explain concepts simply for college students.
- Use bullet points for clarity.
- Be comprehensive but concise.

Course Material:
{context}

Student Question:
{question}

Answer:
"""


    # -----------------------------------------
    # Gemini
    # -----------------------------------------

    llm = get_llm()

    response = llm.invoke(
        prompt
    )


    return (
        response.content,
        docs
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    # -----------------------------------------
    # BRAND
    # -----------------------------------------

    st.markdown(
        """
        <div style="
            padding: 16px;
            background: linear-gradient(
                135deg,
                rgba(124, 58, 237, 0.15) 0%,
                rgba(139, 92, 246, 0.08) 100%
            );
            border-radius: 14px;
            margin-bottom: 24px;
            border: 1px solid rgba(124, 58, 237, 0.2);
        ">

            <div style="
                font-size: 32px;
                font-weight: 800;
                color: #F8F7FF;
                margin-bottom: 8px;
                letter-spacing: -0.5px;
            ">
                📚 Course-Mate
            </div>

            <div style="
                color: #A78BFA;
                font-size: 14px;
                font-weight: 500;
            ">
                Your AI Study Companion
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.divider()


    # -----------------------------------------
    # STUDY MATERIAL
    # -----------------------------------------

    st.markdown(
        "### 📄 Study Material"
    )


    uploaded_file = st.file_uploader(

        "Upload your course PDF",

        type=["pdf"],

        label_visibility="collapsed",

        help=(
            "Upload a PDF file containing "
            "your course notes"
        )
    )


    if uploaded_file:

        # -----------------------------------------
        # Detect new document
        # -----------------------------------------

        if (
            st.session_state.document_name
            != uploaded_file.name
        ):

            st.session_state.processed = False


        # -----------------------------------------
        # File information
        # -----------------------------------------

        st.markdown(

            f"""
            <div style="
                margin-top: 12px;
                padding: 14px;
                background: linear-gradient(
                    135deg,
                    rgba(124, 58, 237, 0.12) 0%,
                    rgba(139, 92, 246, 0.06) 100%
                );
                border: 1px solid rgba(124, 58, 237, 0.25);
                border-radius: 12px;
            ">

                <div style="
                    color: #F5F3FF;
                    font-size: 14px;
                    font-weight: 700;
                ">
                    📄 {uploaded_file.name}
                </div>

                <div style="
                    color: #8E849B;
                    font-size: 12px;
                    margin-top: 6px;
                ">
                    {
                        uploaded_file.size /
                        (1024 * 1024)
                    :.2f} MB
                </div>

            </div>
            """,

            unsafe_allow_html=True
        )


        st.write("")


        # -----------------------------------------
        # PROCESS BUTTON
        # -----------------------------------------

        if st.button(

            "⚡ Process PDF",

            use_container_width=True,

            help=(
                "Process the PDF to create "
                "a searchable knowledge base"
            )
        ):

            with st.spinner(
                "📖 Reading and indexing your material..."
            ):

                try:

                    pages, chunks = process_pdf(
                        uploaded_file
                    )


                    st.success(
                        f"✅ Processed {pages} pages "
                        f"into {chunks} searchable chunks."
                    )


                    st.rerun()


                except Exception as e:

                    st.error(
                        f"❌ Processing failed: {str(e)}"
                    )


    st.divider()


    # -----------------------------------------
    # KNOWLEDGE BASE
    # -----------------------------------------

    st.markdown(
        "### 🧠 Knowledge Base"
    )


    if st.session_state.processed:

        st.markdown(

            """
            <div class="status-card">

                <div class="status-title">
                    🟢 Connected & Ready
                </div>

                <div class="status-desc">
                    Your notes are indexed and ready
                    for questions.
                </div>

            </div>
            """,

            unsafe_allow_html=True
        )


        st.write("")


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "📑 Pages",
                st.session_state.document_pages
            )


        with col2:

            st.metric(
                "🔗 Chunks",
                st.session_state.document_chunks
            )


        st.write("")


        st.caption(
            f"**Document:** "
            f"{st.session_state.document_name}"
        )


        st.write("")


        # -----------------------------------------
        # CLEAR DATABASE
        # -----------------------------------------

        if st.button(

            "🗑️ Clear Knowledge Base",

            use_container_width=True,

            help="Remove the current knowledge base"
        ):

            st.session_state.vectorstore = None

            st.session_state.document_name = None

            st.session_state.document_pages = 0

            st.session_state.document_chunks = 0

            st.session_state.processed = False

            st.session_state.messages = []


            if CHROMA_DIR.exists():

                shutil.rmtree(
                    CHROMA_DIR,
                    ignore_errors=True
                )


            st.success(
                "✅ Knowledge base cleared."
            )

            st.rerun()


    else:

        st.markdown(

            """
            <div class="info-card">

                <div class="info-text">
                    📌 Upload and process a PDF
                    to create your searchable
                    knowledge base.
                </div>

            </div>
            """,

            unsafe_allow_html=True
        )


# =========================================================
# MAIN AREA - HEADER
# =========================================================

st.markdown(

    """
    <div style="
        text-align: center;
        padding: 40px 20px;
    ">

        <div style="
            font-size: 3.5rem;
            font-weight: 800;
            color: #F8F7FF;
            letter-spacing: -1px;

            background: linear-gradient(
                135deg,
                #F8F7FF 0%,
                #A78BFA 100%
            );

            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;

            margin-bottom: 16px;
        ">
            Ask Your Notes
        </div>

        <div style="
            font-size: 16px;
            color: #82768F;
            max-width: 600px;
            margin: 0 auto;
            line-height: 1.8;
        ">
            Upload your course materials and get
            instant answers powered by AI.
            Study smarter, not harder. 🚀
        </div>

    </div>
    """,

    unsafe_allow_html=True
)


# =========================================================
# EMPTY STATE
# =========================================================

if (
    not st.session_state.messages
    and not st.session_state.processed
):

    col1, col2, col3 = st.columns(
        3,
        gap="medium"
    )


    # -----------------------------------------
    # CARD 1
    # -----------------------------------------

    with col1:

        st.markdown(

            """
            <div class="feature-card">

                <div class="feature-icon">
                    📄
                </div>

                <div class="feature-title">
                    Upload Notes
                </div>

                <div class="feature-desc">
                    Select your course PDF and
                    upload it to get started.
                </div>

            </div>
            """,

            unsafe_allow_html=True
        )


    # -----------------------------------------
    # CARD 2
    # -----------------------------------------

    with col2:

        st.markdown(

            """
            <div class="feature-card">

                <div class="feature-icon">
                    ⚡
                </div>

                <div class="feature-title">
                    Process & Index
                </div>

                <div class="feature-desc">
                    Click process to analyze and
                    index your course material.
                </div>

            </div>
            """,

            unsafe_allow_html=True
        )


    # -----------------------------------------
    # CARD 3
    # -----------------------------------------

    with col3:

        st.markdown(

            """
            <div class="feature-card">

                <div class="feature-icon">
                    🎓
                </div>

                <div class="feature-title">
                    Ask Questions
                </div>

                <div class="feature-desc">
                    Chat with your notes to
                    understand concepts better.
                </div>

            </div>
            """,

            unsafe_allow_html=True
        )


elif (
    not st.session_state.messages
    and st.session_state.processed
):

    st.markdown(

        """
        <div style="
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #766B84;
            font-size: 15px;
        ">
            ✨ Your knowledge base is ready.
            Ask your first question below!
        </div>
        """,

        unsafe_allow_html=True
    )


# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(

        message["role"],

        avatar=(
            "👤"
            if message["role"] == "user"
            else "🤖"
        )
    ):

        st.markdown(
            message["content"]
        )


        # -----------------------------------------
        # SOURCES
        # -----------------------------------------

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                "📚 View Sources",
                expanded=False
            ):

                shown_pages = set()


                for doc in message["sources"]:

                    page = (
                        doc.metadata.get(
                            "page",
                            0
                        )
                        + 1
                    )


                    if page not in shown_pages:

                        st.markdown(
                            f"**Page {page}**"
                        )

                        shown_pages.add(
                            page
                        )


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(

    "Ask something about your notes...",

    max_chars=1000
)


# =========================================================
# HANDLE QUESTION
# =========================================================

if question:

    # -----------------------------------------
    # PDF NOT PROCESSED
    # -----------------------------------------

    if not st.session_state.processed:

        st.warning(

            "⚠️ Please upload and process "
            "your course PDF first.",

            icon="📄"
        )


    # -----------------------------------------
    # ASK QUESTION
    # -----------------------------------------

    else:

        # -----------------------------------------
        # Save user message
        # -----------------------------------------

        st.session_state.messages.append({

            "role": "user",

            "content": question

        })


        # -----------------------------------------
        # Display user message
        # -----------------------------------------

        with st.chat_message(
            "user",
            avatar="👤"
        ):

            st.markdown(
                question
            )


        # -----------------------------------------
        # Generate answer
        # -----------------------------------------

        with st.chat_message(
            "assistant",
            avatar="🤖"
        ):

            with st.spinner(
                "🔍 Searching your notes..."
            ):

                try:

                    answer, sources = (
                        ask_course_mate(
                            question
                        )
                    )


                    # ---------------------------------
                    # Show answer
                    # ---------------------------------

                    st.markdown(
                        answer
                    )


                    # ---------------------------------
                    # Show sources
                    # ---------------------------------

                    if sources:

                        with st.expander(
                            "📚 View Sources",
                            expanded=False
                        ):

                            shown_pages = set()


                            for doc in sources:

                                page = (
                                    doc.metadata.get(
                                        "page",
                                        0
                                    )
                                    + 1
                                )


                                if page not in shown_pages:

                                    st.markdown(
                                        f"**Page {page}**"
                                    )

                                    shown_pages.add(
                                        page
                                    )


                    # ---------------------------------
                    # Save assistant message
                    # ---------------------------------

                    st.session_state.messages.append({

                        "role": "assistant",

                        "content": answer,

                        "sources": sources

                    })


                except Exception as e:

                    error_message = (
                        f"❌ Something went wrong: "
                        f"{str(e)}"
                    )


                    st.error(
                        error_message
                    )


                    st.session_state.messages.append({

                        "role": "assistant",

                        "content": error_message,

                        "sources": []

                    })