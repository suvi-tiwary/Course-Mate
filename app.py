import os
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

import numpy as np


# =========================================================
# CONFIG
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error(
        "🔑 GEMINI_API_KEY is missing.\n\n"
        "Add it to your .env file and restart the app."
    )
    st.stop()


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

    * {
        font-family:
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            Roboto,
            Oxygen,
            Ubuntu,
            Cantarell,
            sans-serif;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 20% 0%,
                rgba(124, 58, 237, 0.08),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #08070D 0%,
                #0a0913 100%
            ) !important;

        color: #F5F3FF !important;
    }

    .main {
        background: transparent !important;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem !important;
        padding-bottom: 7rem !important;
    }

    header[data-testid="stHeader"] {
        background: rgba(8, 7, 13, 0.85) !important;
        backdrop-filter: blur(12px) !important;
        border-bottom: 1px solid #181521 !important;
    }

    header[data-testid="stHeader"] > div {
        background: transparent !important;
    }

    header[data-testid="stHeader"] button {
        color: #9B8FB8 !important;
    }

    header[data-testid="stHeader"] button:hover {
        color: #A78BFA !important;
    }

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                rgba(12, 10, 18, 0.98),
                rgba(9, 8, 14, 0.98)
            ) !important;

        border-right: 1px solid #211C2C !important;
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

    h1,
    h2,
    h3 {
        color: #F8F7FF !important;
        letter-spacing: -0.5px;
        font-weight: 700 !important;
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
        line-height: 1.6 !important;
    }

    .stButton > button {
        background:
            linear-gradient(
                135deg,
                #15111F 0%,
                #1a1425 100%
            ) !important;

        color: #EDE9FE !important;

        border: 1px solid #332A46 !important;

        border-radius: 12px !important;

        font-weight: 600 !important;

        padding: 0.6rem 1.2rem !important;

        transition: all 0.25s ease !important;
    }

    .stButton > button:hover {
        background:
            linear-gradient(
                135deg,
                #21182F 0%,
                #2a1d3a 100%
            ) !important;

        border-color: #8B5CF6 !important;

        color: white !important;

        box-shadow:
            0 0 24px rgba(139, 92, 246, 0.25) !important;

        transform: translateY(-2px) !important;
    }

    [data-testid="stFileUploader"] {
        background: rgba(16, 13, 23, 0.65) !important;

        border: 2px dashed #393047 !important;

        border-radius: 16px !important;

        padding: 1.5rem !important;

        transition: all 0.25s ease !important;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #7C3AED !important;

        background:
            rgba(25, 18, 38, 0.9) !important;
    }

    [data-testid="stFileUploader"] section {
        background: transparent !important;
    }

    [data-testid="stFileUploader"] small {
        color: #81768F !important;
    }

    [data-testid="stFileUploaderFile"] {
        background:
            linear-gradient(
                135deg,
                #17131F 0%,
                #1c1725 100%
            ) !important;

        border: 1px solid #292332 !important;

        border-radius: 12px !important;
    }

    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;

        animation: fadeIn 0.3s ease-out;
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
        background: rgba(124, 58, 237, 0.12) !important;
        color: #C4B5FD !important;
        padding: 0.2rem 0.5rem !important;
        border-radius: 6px !important;
    }

    [data-testid="stChatMessage"] pre {
        background: rgba(16, 13, 23, 0.85) !important;
        border: 1px solid #211B2D !important;
        border-radius: 10px !important;
    }

    [data-testid="stChatMessage"]:has(
        [data-testid="chatAvatarIcon-user"]
    ) {
        background:
            rgba(16, 13, 24, 0.65) !important;

        border:
            1px solid #211B2D !important;

        border-radius: 16px !important;

        padding:
            1rem 1.2rem !important;

        margin-bottom:
            1rem !important;
    }

    [data-testid="stChatInput"] {
        background:
            rgba(15, 12, 22, 0.95) !important;

        backdrop-filter: blur(12px) !important;

        border:
            1.5px solid #30283D !important;

        border-radius:
            16px !important;

        box-shadow:
            0 8px 32px rgba(0, 0, 0, 0.35) !important;
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
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #766B84 !important;
        opacity: 0.8 !important;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color:
            #7C3AED !important;

        box-shadow:
            0 0 0 2px rgba(124, 58, 237, 0.15),
            0 12px 40px rgba(124, 58, 237, 0.15) !important;
    }

    [data-testid="stChatInput"] button {
        background:
            linear-gradient(
                135deg,
                #6D28D9 0%,
                #7C3AED 100%
            ) !important;

        color: white !important;

        border: none !important;

        border-radius: 10px !important;
    }

    [data-testid="stChatInput"] button:hover {
        background:
            linear-gradient(
                135deg,
                #7C3AED 0%,
                #8B5CF6 100%
            ) !important;

        box-shadow:
            0 6px 20px rgba(124, 58, 237, 0.3) !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background:
            rgba(16, 13, 23, 0.5) !important;

        border:
            1px solid #211B2D !important;

        border-radius:
            16px !important;
    }

    details {
        border:
            1px solid #211B2D !important;

        border-radius:
            12px !important;

        background:
            rgba(16, 13, 23, 0.5) !important;
    }

    summary {
        color:
            #F5F3FF !important;

        font-weight:
            600 !important;
    }

    hr {
        border-color:
            #211B2D !important;
    }

    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: transparent;
    }

    ::-webkit-scrollbar-thumb {
        background:
            rgba(43, 36, 54, 0.8);

        border-radius:
            10px;
    }

    @keyframes fadeIn {

        from {
            opacity: 0;
            transform: translateY(8px);
        }

        to {
            opacity: 1;
            transform: translateY(0);
        }

    }

    @media (max-width: 768px) {

        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        h1 {
            font-size: 1.8rem !important;
        }

        h2 {
            font-size: 1.4rem !important;
        }

        [data-testid="stChatInput"] {
            border-radius: 12px !important;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# PATHS
# =========================================================

TEMP_DIR = "./temp"

Path(TEMP_DIR).mkdir(
    exist_ok=True
)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents" not in st.session_state:
    st.session_state.documents = []

if "embeddings_matrix" not in st.session_state:
    st.session_state.embeddings_matrix = None

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
# COSINE SIMILARITY
# =========================================================

def cosine_similarity(query_vector, matrix):

    query_vector = np.asarray(
        query_vector,
        dtype=np.float32
    )

    matrix = np.asarray(
        matrix,
        dtype=np.float32
    )

    query_norm = np.linalg.norm(
        query_vector
    )

    matrix_norm = np.linalg.norm(
        matrix,
        axis=1
    )

    denominator = (
        matrix_norm * query_norm
    )

    denominator[
        denominator == 0
    ] = 1e-10

    scores = (
        matrix @ query_vector
    ) / denominator

    return scores


# =========================================================
# MMR SEARCH
# =========================================================

def mmr_search(
    query_vector,
    document_vectors,
    documents,
    k=4,
    fetch_k=10,
    lambda_mult=0.5
):

    if len(documents) == 0:
        return []

    query_vector = np.asarray(
        query_vector,
        dtype=np.float32
    )

    document_vectors = np.asarray(
        document_vectors,
        dtype=np.float32
    )

    query_scores = cosine_similarity(
        query_vector,
        document_vectors
    )

    fetch_k = min(
        fetch_k,
        len(documents)
    )

    candidate_indices = list(
        np.argsort(
            query_scores
        )[::-1][:fetch_k]
    )

    selected = []

    while (
        len(selected) < k
        and candidate_indices
    ):

        if not selected:

            best = candidate_indices[0]

        else:

            mmr_scores = []

            for idx in candidate_indices:

                relevance = query_scores[idx]

                selected_vectors = (
                    document_vectors[selected]
                )

                diversity_scores = cosine_similarity(
                    document_vectors[idx],
                    selected_vectors
                )

                diversity = max(
                    diversity_scores
                )

                score = (
                    lambda_mult * relevance
                    -
                    (1 - lambda_mult)
                    * diversity
                )

                mmr_scores.append(
                    (score, idx)
                )

            best = max(
                mmr_scores,
                key=lambda x: x[0]
            )[1]

        selected.append(best)

        candidate_indices.remove(best)

    return [
        documents[i]
        for i in selected
    ]


# =========================================================
# LOAD DOCUMENT
# =========================================================

def load_uploaded_document(
    uploaded_file
):

    file_path = os.path.join(
        TEMP_DIR,
        uploaded_file.name
    )

    with open(
        file_path,
        "wb"
    ) as f:

        f.write(
            uploaded_file.getbuffer()
        )

    extension = Path(
        uploaded_file.name
    ).suffix.lower()

    if extension == ".pdf":

        loader = PyPDFLoader(
            file_path
        )

    elif extension == ".txt":

        loader = TextLoader(
            file_path,
            encoding="utf-8"
        )

    else:

        raise ValueError(
            "Only PDF and TXT files are supported."
        )

    documents = loader.load()

    return documents


# =========================================================
# PROCESS DOCUMENT
# =========================================================

def process_document(uploaded_file):

    documents = load_uploaded_document(
        uploaded_file
    )

    if not documents:

        raise ValueError(
            "No readable content was found."
        )

    # -----------------------------------------
    # Text splitting
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
            "No text chunks were created."
        )

    # -----------------------------------------
    # Embeddings
    # -----------------------------------------

    embeddings = get_embeddings()

    texts = [
        doc.page_content
        for doc in chunks
    ]

    vectors = embeddings.embed_documents(
        texts
    )

    vectors = np.asarray(
        vectors,
        dtype=np.float32
    )

    # -----------------------------------------
    # Save
    # -----------------------------------------

    st.session_state.documents = chunks

    st.session_state.embeddings_matrix = vectors

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

    documents = (
        st.session_state.documents
    )

    vectors = (
        st.session_state.embeddings_matrix
    )

    if not documents or vectors is None:

        return (
            "Please upload and process your "
            "course material first.",
            []
        )

    embeddings = get_embeddings()

    # -----------------------------------------
    # Embed question
    # -----------------------------------------

    query_vector = embeddings.embed_query(
        question
    )

    # -----------------------------------------
    # MMR retrieval
    # -----------------------------------------

    docs = mmr_search(
        query_vector=query_vector,
        document_vectors=vectors,
        documents=documents,
        k=4,
        fetch_k=10,
        lambda_mult=0.5
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
            doc.metadata.get(
                "page",
                0
            ) + 1
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
You are Course-Mate, an AI study assistant
designed to help students learn better.

Answer the student's question using ONLY
the provided course material.

Guidelines:

- Use only the provided material.
- Do not make up information.
- Do not use outside knowledge.
- If the answer is not available in the material,
  clearly say that.
- Explain concepts in simple language.
- Use bullet points or numbered lists when useful.
- Give examples when they are supported by the notes.
- Be concise but comprehensive.
- Make the answer easy to revise for exams.

COURSE MATERIAL:

{context}

STUDENT QUESTION:

{question}

ANSWER:
"""

    # -----------------------------------------
    # Gemini
    # -----------------------------------------

    llm = get_llm()

    response = llm.invoke(
        prompt
    )

    answer = response.content

    return (
        answer,
        docs
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            padding: 18px;
            background:
                linear-gradient(
                    135deg,
                    rgba(124,58,237,0.14),
                    rgba(139,92,246,0.04)
                );
            border: 1px solid #211B2D;
            border-radius: 14px;
            margin-bottom: 18px;
        ">

            <div style="
                font-size: 27px;
                font-weight: 800;
                color: #F8F7FF;
            ">
                📚 Course-Mate
            </div>

            <div style="
                margin-top: 7px;
                color: #A78BFA;
                font-size: 13px;
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
    # Study Material
    # -----------------------------------------

    st.markdown(
        "### 📄 Study Material"
    )

    uploaded_file = st.file_uploader(
        "Upload your course material",
        type=["pdf", "txt"],
        label_visibility="collapsed",
        help="Upload PDF or TXT course notes."
    )

    if uploaded_file:

        if (
            st.session_state.document_name
            != uploaded_file.name
        ):

            st.session_state.processed = False

        st.markdown(
            f"""
            <div style="
                margin-top: 12px;
                padding: 14px;

                background:
                    linear-gradient(
                        135deg,
                        rgba(109,40,217,0.12),
                        rgba(124,58,237,0.04)
                    );

                border:
                    1px solid rgba(124,58,237,0.3);

                border-radius: 12px;
            ">

                <div style="
                    color: #F5F3FF;
                    font-size: 13px;
                    font-weight: 600;
                    word-break: break-word;
                ">
                    📄 {uploaded_file.name}
                </div>

                <div style="
                    color: #A8A0B8;
                    font-size: 11px;
                    margin-top: 6px;
                ">
                    {uploaded_file.size / (1024 * 1024):.2f} MB
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        if st.button(
            "⚡ Process Material",
            use_container_width=True
        ):

            with st.spinner(
                "📖 Reading and indexing your material..."
            ):

                try:

                    pages, chunks = (
                        process_document(
                            uploaded_file
                        )
                    )

                    st.success(
                        f"✅ Processed {pages} pages "
                        f"into {chunks} searchable chunks."
                    )

                    st.rerun()

                except Exception as e:

                    st.error(
                        f"❌ Processing failed:\n\n{str(e)}"
                    )

    st.divider()

    # -----------------------------------------
    # Knowledge Base
    # -----------------------------------------

    st.markdown(
        "### 🧠 Knowledge Base"
    )

    if st.session_state.processed:

        st.markdown(
            """
            <div style="
                padding: 14px;

                background:
                    linear-gradient(
                        135deg,
                        rgba(16,65,48,0.2),
                        rgba(22,68,59,0.1)
                    );

                border:
                    1px solid rgba(134,239,172,0.3);

                border-radius: 12px;
            ">

                <div style="
                    color: #86EFAC;
                    font-weight: 700;
                    font-size: 14px;
                ">
                    🟢 Connected & Ready
                </div>

                <div style="
                    color: #6EE7B7;
                    font-size: 12px;
                    margin-top: 6px;
                ">
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

        if st.button(
            "🗑️ Clear Knowledge Base",
            use_container_width=True
        ):

            st.session_state.documents = []

            st.session_state.embeddings_matrix = None

            st.session_state.document_name = None

            st.session_state.document_pages = 0

            st.session_state.document_chunks = 0

            st.session_state.processed = False

            st.session_state.messages = []

            # Delete temporary documents
            if os.path.exists(TEMP_DIR):

                shutil.rmtree(
                    TEMP_DIR
                )

            Path(TEMP_DIR).mkdir(
                exist_ok=True
            )

            st.success(
                "✅ Knowledge base cleared."
            )

            st.rerun()

    else:

        st.markdown(
            """
            <div style="
                padding: 14px;

                background:
                    rgba(16,13,23,0.5);

                border:
                    1px solid #211B2D;

                border-radius: 12px;
            ">

                <div style="
                    color: #8E849B;
                    font-size: 13px;
                    line-height: 1.6;
                ">

                    📌 Upload and process a PDF
                    or TXT file to create your
                    searchable knowledge base.

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    """
    <div style="
        text-align: center;
        padding: 40px 20px 30px 20px;
    ">

        <div style="
            font-size: 3rem;
            font-weight: 800;

            background:
                linear-gradient(
                    135deg,
                    #F8F7FF 0%,
                    #A78BFA 100%
                );

            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;

            background-clip: text;

            letter-spacing: -1px;
        ">
            Ask Your Notes
        </div>

        <div style="
            margin-top: 16px;

            color: #82768F;

            font-size: 17px;

            max-width: 600px;

            margin-left: auto;
            margin-right: auto;

            line-height: 1.7;
        ">
            Upload your course materials and
            get instant answers powered by AI.
            <br>
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

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div style="
                padding: 24px 20px;

                background:
                    rgba(16,13,23,0.5);

                border:
                    1px solid #211B2D;

                border-radius: 14px;

                min-height: 160px;
            ">

                <div style="
                    font-size: 32px;
                ">
                    📄
                </div>

                <div style="
                    margin-top: 12px;

                    color: #F5F3FF;

                    font-weight: 700;

                    font-size: 15px;
                ">
                    Upload Notes
                </div>

                <div style="
                    margin-top: 8px;

                    color: #8E849B;

                    font-size: 13px;

                    line-height: 1.6;
                ">
                    Select your course PDF or TXT
                    and upload it to get started.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div style="
                padding: 24px 20px;

                background:
                    rgba(16,13,23,0.5);

                border:
                    1px solid #211B2D;

                border-radius: 14px;

                min-height: 160px;
            ">

                <div style="
                    font-size: 32px;
                ">
                    ⚡
                </div>

                <div style="
                    margin-top: 12px;

                    color: #F5F3FF;

                    font-weight: 700;

                    font-size: 15px;
                ">
                    Process & Index
                </div>

                <div style="
                    margin-top: 8px;

                    color: #8E849B;

                    font-size: 13px;

                    line-height: 1.6;
                ">
                    Process your material and create
                    searchable semantic embeddings.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div style="
                padding: 24px 20px;

                background:
                    rgba(16,13,23,0.5);

                border:
                    1px solid #211B2D;

                border-radius: 14px;

                min-height: 160px;
            ">

                <div style="
                    font-size: 32px;
                ">
                    🎓
                </div>

                <div style="
                    margin-top: 12px;

                    color: #F5F3FF;

                    font-weight: 700;

                    font-size: 15px;
                ">
                    Ask Questions
                </div>

                <div style="
                    margin-top: 8px;

                    color: #8E849B;

                    font-size: 13px;

                    line-height: 1.6;
                ">
                    Chat with your notes and get
                    context-based answers.
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
                        ) + 1
                    )

                    if page not in shown_pages:

                        st.markdown(
                            f"**📄 Page {page}**"
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

    if not st.session_state.processed:

        st.warning(
            "⚠️ Please upload and process "
            "your course material first.",
            icon="📄"
        )

    else:

        # -----------------------------------------
        # User message
        # -----------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message(
            "user",
            avatar="👤"
        ):

            st.markdown(
                question
            )

        # -----------------------------------------
        # AI response
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

                    st.markdown(
                        answer
                    )

                    # ---------------------------------
                    # Sources
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
                                    ) + 1
                                )

                                if page not in shown_pages:

                                    st.markdown(
                                        f"**📄 Page {page}**"
                                    )

                                    shown_pages.add(
                                        page
                                    )

                    # ---------------------------------
                    # Save AI message
                    # ---------------------------------

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        }
                    )

                except Exception as e:

                    error_message = (
                        "❌ Something went wrong.\n\n"
                        f"`{str(e)}`"
                    )

                    st.error(
                        error_message
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_message,
                            "sources": []
                        }
                    )