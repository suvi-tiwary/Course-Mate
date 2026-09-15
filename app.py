import os
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI


# =========================================================
# CONFIG
# =========================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY is missing. Add it to your .env file.")
    st.stop()

st.set_page_config(
    page_title="Course-Mate",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* =========================
       GLOBAL
    ========================= */

    html,
    body,
    [data-testid="stAppViewContainer"],
    [data-testid="stApp"] {
        background: #080611 !important;
        color: #f5f1ff !important;
    }

    [data-testid="stHeader"] {
        background: #080611 !important;
        border-bottom: 1px solid #211a32 !important;
    }

    [data-testid="stToolbar"] {
        background: transparent !important;
    }

    .main {
        background: #080611 !important;
    }

    /* Remove Streamlit default top spacing */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 7rem !important;
        max-width: 1400px !important;
    }


    /* =========================
       SIDEBAR
    ========================= */

    section[data-testid="stSidebar"] {
        background:
            radial-gradient(
                circle at 30% 10%,
                rgba(121, 48, 190, 0.20),
                transparent 35%
            ),
            #0c0915 !important;

        border-right: 1px solid #211a32 !important;
    }

    section[data-testid="stSidebar"] > div {
        background: transparent !important;
    }

    .sidebar-brand {
        padding: 18px 10px 28px 10px;
    }

    .brand-title {
        font-size: 27px;
        font-weight: 800;
        color: #f8f7ff;
        letter-spacing: -1px;
    }

    .brand-subtitle {
        margin-top: 7px;
        color: #a78bfa;
        font-size: 13px;
        font-weight: 500;
    }

    .sidebar-section {
        margin-top: 30px;
        padding: 0 8px;
    }

    .sidebar-heading {
        color: #f1ecff;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 14px;
    }


    /* =========================
       HERO
    ========================= */

    .hero {
        min-height: 330px;

        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;

        text-align: center;

        padding: 35px 20px;

        background:
            radial-gradient(
                circle at 50% 35%,
                rgba(124, 58, 237, 0.22),
                transparent 38%
            ),
            linear-gradient(
                180deg,
                #0d0918 0%,
                #080611 100%
            );

        border-radius: 28px;
        border: 1px solid #211a32;

        box-shadow:
            0 20px 80px rgba(0,0,0,0.35);
    }

    .hero-icon {
        width: 74px;
        height: 74px;

        display: flex;
        align-items: center;
        justify-content: center;

        border-radius: 22px;

        background:
            linear-gradient(
                135deg,
                #7c3aed,
                #a855f7
            );

        box-shadow:
            0 15px 45px rgba(124,58,237,0.35);

        font-size: 35px;

        margin-bottom: 22px;
    }

    .hero-title {
        font-size: clamp(2.4rem, 5vw, 4rem);
        font-weight: 800;

        margin: 0;

        letter-spacing: -2px;

        background:
            linear-gradient(
                135deg,
                #ffffff 0%,
                #ddd0ff 45%,
                #a78bfa 100%
            );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .hero-text {
        max-width: 650px;

        margin-top: 15px;

        color: #9d94ad;

        font-size: 16px;
        line-height: 1.7;
    }


    /* =========================
       UPLOAD CARD
    ========================= */

    .upload-title {
        margin-top: 28px;
        margin-bottom: 10px;

        font-size: 20px;
        font-weight: 700;

        color: #f4efff;
    }

    .upload-subtitle {
        color: #837b91;
        font-size: 14px;
        margin-bottom: 12px;
    }

    [data-testid="stFileUploader"] {
        background: #100c1b !important;
        border: 1px dashed #493660 !important;
        border-radius: 18px !important;
        padding: 10px !important;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #8b5cf6 !important;
    }

    [data-testid="stFileUploader"] section {
        background: transparent !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #100c1b !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #a9a0b8 !important;
    }


    /* =========================
       BUTTONS
    ========================= */

    .stButton > button {
        border-radius: 12px !important;

        border: 1px solid #3b2b52 !important;

        background: #171122 !important;

        color: #eee8ff !important;

        font-weight: 600 !important;

        transition: 0.2s ease !important;
    }

    .stButton > button:hover {
        border-color: #8b5cf6 !important;

        background: #211633 !important;

        color: white !important;
    }


    /* =========================
       TEXT INPUT
    ========================= */

    [data-testid="stTextInput"] input {
        background: #15111d !important;

        color: #f4efff !important;

        border: 1px solid #352745 !important;

        border-radius: 14px !important;

        padding: 15px 17px !important;

        font-size: 15px !important;
    }

    [data-testid="stTextInput"] input:focus {
        border-color: #8b5cf6 !important;

        box-shadow:
            0 0 0 1px #8b5cf6 !important;
    }

    [data-testid="stTextInput"] input::placeholder {
        color: #756b82 !important;
    }


    /* =========================
       ANSWER CARD
    ========================= */

    .answer-card {
        margin-top: 25px;

        padding: 24px;

        background: #100c19;

        border: 1px solid #292039;

        border-radius: 20px;

        box-shadow:
            0 15px 50px rgba(0,0,0,0.25);
    }

    .answer-label {
        color: #a78bfa;

        font-size: 13px;

        font-weight: 700;

        text-transform: uppercase;

        letter-spacing: 1px;

        margin-bottom: 12px;
    }


    /* =========================
       FILE STATUS
    ========================= */

    .file-status {
        margin-top: 15px;

        padding: 14px 16px;

        border-radius: 12px;

        background: #120e1c;

        border: 1px solid #292039;

        color: #c8bfd4;

        font-size: 14px;
    }


    /* =========================
       DIVIDER
    ========================= */

    hr {
        border-color: #211a32 !important;
    }


    /* =========================
       MOBILE
    ========================= */

    @media (max-width: 768px) {

        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        .hero {
            min-height: 280px;
            padding: 25px 15px;
        }

        .hero-title {
            font-size: 2.5rem;
        }

        .hero-text {
            font-size: 14px;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SESSION STATE
# =========================================================

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None

if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-title">📚 Course-Mate</div>
            <div class="brand-subtitle">
                Your AI Study Companion
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown(
        """
        <div class="sidebar-section">
            <div class="sidebar-heading">
                📄 Study Material
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload your notes",
        type=["pdf", "txt", "docx"],
        label_visibility="collapsed",
    )

    st.caption("Supported: PDF, TXT, DOCX")

    if st.session_state.file_name:
        st.markdown(
            f"""
            <div class="file-status">
                📎 <b>{st.session_state.file_name}</b><br>
                <span style="color:#81768f;">
                Ready for questions
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.caption("Course-Mate • RAG Study Assistant")
    st.caption("Ask questions directly from your notes.")


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-icon">
            📚
        </div>

        <h1 class="hero-title">
            Course-Mate
        </h1>

        <div class="hero-text">
            Turn your study material into an AI-powered
            learning companion. Upload your notes and ask
            questions using RAG.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD DOCUMENT
# =========================================================

def load_document(file_path):

    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":
        loader = PyPDFLoader(file_path)

    elif extension == ".txt":
        loader = TextLoader(
            file_path,
            encoding="utf-8"
        )

    elif extension == ".docx":
        loader = Docx2txtLoader(file_path)

    else:
        raise ValueError("Unsupported file type.")

    return loader.load()


# =========================================================
# CREATE VECTOR DATABASE
# =========================================================

def create_vectorstore(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    persist_directory = "./chroma_db"

    # Remove old database
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    return vectorstore


# =========================================================
# PROCESS UPLOADED FILE
# =========================================================

if uploaded_file is not None:

    if uploaded_file.name != st.session_state.file_name:

        with st.spinner("Processing your study material..."):

            try:

                os.makedirs("documents", exist_ok=True)

                file_path = os.path.join(
                    "documents",
                    uploaded_file.name
                )

                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                documents = load_document(file_path)

                vectorstore = create_vectorstore(documents)

                st.session_state.vectorstore = vectorstore
                st.session_state.file_name = uploaded_file.name
                st.session_state.documents_loaded = True

                st.success(
                    f"✅ {uploaded_file.name} is ready!"
                )

            except Exception as e:

                st.error(
                    f"Processing failed: {str(e)}"
                )


# =========================================================
# QUESTION SECTION
# =========================================================

st.markdown(
    """
    <div class="upload-title">
        Ask Your Notes
    </div>

    <div class="upload-subtitle">
        Ask anything from the uploaded study material.
    </div>
    """,
    unsafe_allow_html=True,
)


question = st.text_input(
    "Question",
    placeholder="Ask something about your notes...",
    label_visibility="collapsed",
)


# =========================================================
# ASK BUTTON
# =========================================================

ask = st.button(
    "✨ Ask Course-Mate",
    use_container_width=True,
)


# =========================================================
# RAG QUESTION ANSWERING
# =========================================================

if ask:

    if not question.strip():

        st.warning("Please enter a question.")

    elif st.session_state.vectorstore is None:

        st.warning(
            "Please upload your study material first."
        )

    else:

        with st.spinner("Thinking from your notes..."):

            try:

                vectorstore = st.session_state.vectorstore

                # Retrieve relevant chunks
                docs = vectorstore.similarity_search(
                    question,
                    k=4
                )

                context = "\n\n".join(
                    doc.page_content
                    for doc in docs
                )

                # Gemini
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.2,
                )

                prompt = f"""
You are Course-Mate, an AI study assistant.

Answer the student's question using the provided
study material.

Rules:
1. Use the study material as the primary source.
2. If the answer is not present in the material,
   clearly say that it is not available in the uploaded notes.
3. Explain difficult concepts simply.
4. Use bullet points when useful.
5. Do not invent information.

STUDY MATERIAL:
{context}

STUDENT QUESTION:
{question}
"""

                response = llm.invoke(prompt)

                answer = response.content

                # Answer UI
                st.markdown(
                    f"""
                    <div class="answer-card">

                        <div class="answer-label">
                            Course-Mate Answer
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(answer)

                # Sources
                with st.expander("📚 View retrieved sources"):

                    for i, doc in enumerate(docs, 1):

                        st.markdown(
                            f"**Source {i}**"
                        )

                        st.write(
                            doc.page_content[:1000]
                        )

                        st.divider()

            except Exception as e:

                st.error(
                    f"Something went wrong: {str(e)}"
                )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div style="
        text-align:center;
        margin-top:60px;
        padding:20px;
        color:#625970;
        font-size:12px;
    ">
        Built with ❤️ using RAG + LangChain + Gemini
    </div>
    """,
    unsafe_allow_html=True,
)