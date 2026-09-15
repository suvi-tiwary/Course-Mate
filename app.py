import os
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


# =========================================================
# CONFIG
# =========================================================

load_dotenv()

st.set_page_config(
    page_title="Course-Mate | AI Study Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# API KEY
# =========================================================

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# =========================================================
# DIRECTORIES
# =========================================================

TEMP_DIR = Path("./temp")
TEMP_DIR.mkdir(exist_ok=True)


# =========================================================
# SESSION STATE
# =========================================================

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "file_name" not in st.session_state:
    st.session_state.file_name = None

if "pages" not in st.session_state:
    st.session_state.pages = 0

if "chunks" not in st.session_state:
    st.session_state.chunks = 0


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* ================================
       GLOBAL
    ================================= */

    .stApp {
        background:
            radial-gradient(
                circle at 15% 10%,
                rgba(124, 58, 237, 0.15),
                transparent 30%
            ),
            radial-gradient(
                circle at 85% 20%,
                rgba(139, 92, 246, 0.10),
                transparent 30%
            ),
            #08070d;
        color: #f5f3ff;
    }

    /* ================================
       SIDEBAR
    ================================= */

    section[data-testid="stSidebar"] {
        background: #0d0b14;
        border-right: 1px solid rgba(139, 92, 246, 0.15);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }

    .brand {
        padding: 10px 5px 25px 5px;
    }

    .brand-title {
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -1px;
        color: #ffffff;
    }

    .brand-title span {
        color: #a855f7;
    }

    .brand-subtitle {
        color: #9ca3af;
        font-size: 13px;
        margin-top: 4px;
    }

    .sidebar-card {
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(168,85,247,0.15);
        border-radius: 16px;
        padding: 18px;
        margin-top: 15px;
    }

    .sidebar-heading {
        color: #e9d5ff;
        font-size: 14px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .status {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 13px;
        color: #c4b5fd;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #a855f7;
        box-shadow: 0 0 10px #a855f7;
    }

    /* ================================
       MAIN HERO
    ================================= */

    .hero {
        padding: 35px 0 25px 0;
    }

    .hero-badge {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 20px;
        background: rgba(168,85,247,0.10);
        border: 1px solid rgba(168,85,247,0.25);
        color: #c084fc;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 15px;
    }

    .hero-title {
        font-size: 48px;
        line-height: 1.08;
        font-weight: 800;
        letter-spacing: -2px;
        margin-bottom: 12px;
        color: #ffffff;
    }

    .hero-title span {
        background: linear-gradient(
            90deg,
            #a855f7,
            #c084fc,
            #e9d5ff
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-text {
        color: #9ca3af;
        font-size: 16px;
        line-height: 1.6;
        max-width: 720px;
    }

    /* ================================
       FEATURE CARDS
    ================================= */

    .feature-card {
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(168,85,247,0.13);
        border-radius: 18px;
        padding: 20px;
        height: 150px;
        transition: 0.2s;
    }

    .feature-icon {
        font-size: 25px;
        margin-bottom: 10px;
    }

    .feature-title {
        color: #f5f3ff;
        font-weight: 700;
        font-size: 15px;
        margin-bottom: 5px;
    }

    .feature-text {
        color: #8f8a9d;
        font-size: 12px;
        line-height: 1.5;
    }

    /* ================================
       CHAT
    ================================= */

    .chat-header {
        margin-top: 35px;
        margin-bottom: 15px;
        font-size: 22px;
        font-weight: 700;
        color: #ffffff;
    }

    /* ================================
       METRICS
    ================================= */

    .metric-card {
        background: rgba(168,85,247,0.06);
        border: 1px solid rgba(168,85,247,0.15);
        border-radius: 14px;
        padding: 12px;
        text-align: center;
    }

    .metric-value {
        color: #c084fc;
        font-size: 22px;
        font-weight: 800;
    }

    .metric-label {
        color: #8f8a9d;
        font-size: 11px;
        margin-top: 2px;
    }

    /* ================================
       BUTTONS
    ================================= */

    .stButton > button {
        border-radius: 10px;
        border: 1px solid rgba(168,85,247,0.25);
        background: #7c3aed;
        color: white;
        font-weight: 600;
        transition: 0.2s;
    }

    .stButton > button:hover {
        border-color: #a855f7;
        background: #8b5cf6;
        color: white;
    }

    /* ================================
       FILE UPLOADER
    ================================= */

    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.025);
        border-radius: 14px;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #12101a !important;
        border: 1px dashed rgba(168,85,247,0.35) !important;
    }

    [data-testid="stFileUploaderDropzone"] * {
        color: #ddd6fe !important;
    }

    /* ================================
       CHAT INPUT
    ================================= */

    [data-testid="stChatInput"] {
        background: #11101a !important;
        border: 1px solid rgba(168,85,247,0.25) !important;
    }

    [data-testid="stChatInput"] textarea {
        background: #11101a !important;
        color: #ffffff !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #77717f !important;
    }

    /* ================================
       EXPANDER
    ================================= */

    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.025);
        border-radius: 10px;
        color: #ddd6fe !important;
    }

    /* ================================
       TEXT
    ================================= */

    p, label {
        color: #d4d0dc;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# CHECK API KEY
# =========================================================

if not GEMINI_API_KEY:
    st.error(
        "❌ GEMINI_API_KEY is missing. "
        "Add it to your .env file locally or Streamlit Secrets after deployment."
    )
    st.stop()


# =========================================================
# EMBEDDINGS
# =========================================================

@st.cache_resource
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# =========================================================
# GEMINI
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

    try:

        # -----------------------------------------
        # Save temporary PDF
        # -----------------------------------------

        file_path = TEMP_DIR / uploaded_file.name

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # -----------------------------------------
        # Load PDF
        # -----------------------------------------

        loader = PyPDFLoader(str(file_path))

        documents = loader.load()

        if not documents:
            raise Exception("No text could be extracted from this PDF.")

        # -----------------------------------------
        # Split into chunks
        # -----------------------------------------

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )

        chunks = splitter.split_documents(documents)

        if not chunks:
            raise Exception("Could not create text chunks from the PDF.")

        # -----------------------------------------
        # Embeddings
        # -----------------------------------------

        embeddings = get_embeddings()

        # -----------------------------------------
        # FREE LOCAL VECTOR DATABASE
        # FAISS
        # -----------------------------------------

        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )

        # -----------------------------------------
        # Save in session
        # -----------------------------------------

        st.session_state.vectorstore = vectorstore
        st.session_state.file_name = uploaded_file.name
        st.session_state.pages = len(documents)
        st.session_state.chunks = len(chunks)
        st.session_state.messages = []

        # -----------------------------------------
        # Remove temporary PDF
        # -----------------------------------------

        try:
            file_path.unlink()
        except Exception:
            pass

        return True, "PDF processed successfully!"

    except Exception as e:

        return False, str(e)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            <div class="brand-title">
                Course<span>-Mate</span>
            </div>
            <div class="brand-subtitle">
                Your AI-powered study companion
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-card">

            <div class="sidebar-heading">
                📚 Knowledge Base
            </div>

        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload your course PDF",
        type=["pdf"],
        label_visibility="collapsed",
    )

    process_button = st.button(
        "⚡ Process PDF",
        use_container_width=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------
    # PROCESS
    # -----------------------------------------

    if process_button:

        if uploaded_file is None:

            st.warning("Please upload a PDF first.")

        else:

            with st.spinner("Reading and indexing your PDF..."):

                success, message = process_pdf(uploaded_file)

            if success:
                st.success(message)
                st.rerun()

            else:
                st.error(f"Processing failed: {message}")

    # -----------------------------------------
    # STATUS
    # -----------------------------------------

    if st.session_state.vectorstore:

        st.markdown(
            """
            <div class="sidebar-card">

                <div class="sidebar-heading">
                    🟢 Knowledge Base Ready
                </div>

                <div class="status">
                    <div class="status-dot"></div>
                    Ready to answer questions
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("###")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">
                        {st.session_state.pages}
                    </div>
                    <div class="metric-label">
                        Pages
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">
                        {st.session_state.chunks}
                    </div>
                    <div class="metric-label">
                        Chunks
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("###")

        st.caption(
            f"📄 {st.session_state.file_name}"
        )

        if st.button(
            "🗑️ Clear Knowledge Base",
            use_container_width=True
        ):

            st.session_state.vectorstore = None
            st.session_state.file_name = None
            st.session_state.pages = 0
            st.session_state.chunks = 0
            st.session_state.messages = []

            st.rerun()

    else:

        st.markdown(
            """
            <div class="sidebar-card">

                <div class="sidebar-heading">
                    ⚪ No Document Loaded
                </div>

                <div class="status">
                    <div class="status-dot"></div>
                    Upload a PDF to begin
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.caption(
        "Course-Mate • RAG Study Assistant"
    )


# =========================================================
# MAIN HERO
# =========================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-badge">
            ✦ RETRIEVAL-AUGMENTED GENERATION
        </div>

        <div class="hero-title">
            Study smarter with
            <span>your own notes.</span>
        </div>

        <div class="hero-text">
            Upload your course material and ask questions.
            Course-Mate searches your notes first and uses
            Gemini to generate grounded answers.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# FEATURE CARDS
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        """
        <div class="feature-card">

            <div class="feature-icon">📄</div>

            <div class="feature-title">
                Upload Notes
            </div>

            <div class="feature-text">
                Upload your course PDF and automatically
                convert it into searchable knowledge.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with col2:

    st.markdown(
        """
        <div class="feature-card">

            <div class="feature-icon">🔎</div>

            <div class="feature-title">
                Smart Retrieval
            </div>

            <div class="feature-text">
                Relevant sections are retrieved from your
                notes before generating an answer.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with col3:

    st.markdown(
        """
        <div class="feature-card">

            <div class="feature-icon">🤖</div>

            <div class="feature-title">
                AI Answers
            </div>

            <div class="feature-text">
                Gemini explains concepts using the
                information found in your study material.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# NO PDF MESSAGE
# =========================================================

if st.session_state.vectorstore is None:

    st.markdown(
        """
        <div style="
            margin-top:45px;
            padding:35px;
            border-radius:20px;
            border:1px solid rgba(168,85,247,0.15);
            background:rgba(255,255,255,0.02);
            text-align:center;
        ">

            <div style="font-size:42px;">
                📚
            </div>

            <div style="
                font-size:20px;
                font-weight:700;
                margin-top:10px;
                color:#ffffff;
            ">
                Upload your course material
            </div>

            <div style="
                color:#8f8a9d;
                font-size:14px;
                margin-top:8px;
            ">
                Upload a PDF from the sidebar to start
                asking questions.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# =========================================================
# CHAT HEADER
# =========================================================

st.markdown(
    """
    <div class="chat-header">
        💬 Ask Your Notes
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# DISPLAY OLD MESSAGES
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# =========================================================
# USER QUESTION
# =========================================================

question = st.chat_input(
    "Ask something from your notes..."
)


if question:

    # -----------------------------------------
    # USER MESSAGE
    # -----------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # -----------------------------------------
    # ASSISTANT
    # -----------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Searching your notes..."):

            try:

                vectorstore = st.session_state.vectorstore

                # -----------------------------------------
                # MMR RETRIEVER
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
                # RETRIEVE
                # -----------------------------------------

                docs = retriever.invoke(question)

                if not docs:

                    answer = (
                        "I couldn't find relevant information "
                        "in your uploaded notes."
                    )

                    st.markdown(answer)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                else:

                    # -----------------------------------------
                    # BUILD CONTEXT
                    # -----------------------------------------

                    context_parts = []

                    for doc in docs:

                        page = doc.metadata.get(
                            "page",
                            "Unknown"
                        )

                        context_parts.append(
                            f"""
                            PAGE: {page + 1 if isinstance(page, int) else page}

                            {doc.page_content}
                            """
                        )

                    context = "\n\n---\n\n".join(
                        context_parts
                    )

                    # -----------------------------------------
                    # PROMPT
                    # -----------------------------------------

                    prompt = f"""
You are Course-Mate, an AI study assistant.

Answer the user's question using ONLY the
provided course material.

COURSE MATERIAL:
{context}

USER QUESTION:
{question}

RULES:

1. Use only the provided course material.
2. Do not invent facts.
3. If the answer is not available in the material,
   clearly say that it is not present in the uploaded notes.
4. Explain the answer clearly for a college student.
5. Use bullet points when useful.
6. For technical concepts, give a simple example
   if the material supports it.
"""

                    # -----------------------------------------
                    # GEMINI
                    # -----------------------------------------

                    llm = get_llm()

                    response = llm.invoke(prompt)

                    answer = response.content

                    # -----------------------------------------
                    # SHOW ANSWER
                    # -----------------------------------------

                    st.markdown(answer)

                    # -----------------------------------------
                    # SOURCES
                    # -----------------------------------------

                    with st.expander(
                        "📚 View sources"
                    ):

                        pages_found = []

                        for doc in docs:

                            page = doc.metadata.get(
                                "page",
                                None
                            )

                            if isinstance(page, int):

                                page_number = page + 1

                                if page_number not in pages_found:

                                    pages_found.append(
                                        page_number
                                    )

                        if pages_found:

                            st.write(
                                "Retrieved from pages: "
                                + ", ".join(
                                    map(str, pages_found)
                                )
                            )

                        for i, doc in enumerate(docs):

                            page = doc.metadata.get(
                                "page",
                                "Unknown"
                            )

                            if isinstance(page, int):

                                page = page + 1

                            st.markdown(
                                f"**Source {i + 1} — Page {page}**"
                            )

                            preview = doc.page_content[:500]

                            st.caption(preview)

                    # -----------------------------------------
                    # SAVE MESSAGE
                    # -----------------------------------------

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

            except Exception as e:

                error_message = (
                    f"Something went wrong while answering: {e}"
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message
                    }
                )