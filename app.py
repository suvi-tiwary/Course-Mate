import os
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

# =========================================================
# CONFIG
# =========================================================

load_dotenv()

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

    /* ================================
       GLOBAL
    ================================= */

    .stApp {
        background:
            radial-gradient(
                circle at 50% 10%,
                rgba(124, 58, 237, 0.10),
                transparent 35%
            ),
            linear-gradient(
                135deg,
                #0b0812 0%,
                #100b1c 50%,
                #08070d 100%
            );

        color: #f5f3ff;
    }

    .main {
        padding-top: 0rem;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    /* Hide default Streamlit decorations */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* ================================
       SIDEBAR
    ================================= */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #0d0915 0%,
                #0a0810 100%
            );

        border-right: 1px solid rgba(139, 92, 246, 0.20);
    }

    section[data-testid="stSidebar"] > div {
        padding: 18px 14px;
    }

    .sidebar-head {
        padding: 18px 16px;
        margin-bottom: 12px;

        border: 1px solid rgba(139, 92, 246, 0.24);
        border-radius: 16px;

        background:
            linear-gradient(
                135deg,
                rgba(124, 58, 237, 0.12),
                rgba(30, 20, 45, 0.40)
            );

        box-shadow:
            0 0 30px rgba(124, 58, 237, 0.05);
    }

    .sidebar-brand {
        font-size: 21px;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.3px;
    }

    .sidebar-subtitle {
        margin-top: 4px;
        font-size: 12px;
        color: #9f9aaf;
    }

    .upload-box {
        padding: 14px;
        margin-top: 10px;

        border: 1px dashed rgba(139, 92, 246, 0.45);
        border-radius: 14px;

        background: rgba(124, 58, 237, 0.035);
    }

    .status-box {
        padding: 14px;
        margin-top: 12px;

        border-radius: 12px;

        background: rgba(124, 58, 237, 0.08);
        border: 1px solid rgba(139, 92, 246, 0.16);
    }

    .status-title {
        font-size: 13px;
        font-weight: 600;
        color: #ddd6fe;
    }

    .status-text {
        font-size: 12px;
        color: #9690a3;
        margin-top: 4px;
    }

    /* ================================
       BUTTON
    ================================= */

    .stButton > button {
        width: 100%;

        border: none;
        border-radius: 10px;

        padding: 11px 15px;

        background:
            linear-gradient(
                135deg,
                #7c3aed,
                #9333ea
            );

        color: white;

        font-weight: 600;

        box-shadow:
            0 8px 25px rgba(124, 58, 237, 0.20);

        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);

        box-shadow:
            0 10px 30px rgba(124, 58, 237, 0.35);
    }

    /* ================================
       HERO
    ================================= */

    .hero {
        max-width: 1050px;
        margin: 80px auto 30px auto;
        text-align: center;
    }

    .hero-badge {
        display: inline-block;

        padding: 7px 14px;

        border-radius: 999px;

        border: 1px solid rgba(139, 92, 246, 0.30);

        background: rgba(124, 58, 237, 0.08);

        color: #c4b5fd;

        font-size: 12px;
        font-weight: 600;

        letter-spacing: 0.8px;
    }

    .hero-title {
        margin-top: 22px;

        font-size: clamp(42px, 6vw, 72px);

        line-height: 1.05;

        font-weight: 800;

        letter-spacing: -2.5px;

        color: #ffffff;
    }

    .hero-title span {
        background:
            linear-gradient(
                90deg,
                #c4b5fd,
                #8b5cf6,
                #d8b4fe
            );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-text {
        max-width: 680px;

        margin: 22px auto 0 auto;

        font-size: 16px;

        line-height: 1.7;

        color: #a9a2b4;
    }

    /* ================================
       FEATURES
    ================================= */

    .features {
        max-width: 1050px;

        margin: 55px auto 30px auto;

        display: grid;

        grid-template-columns:
            repeat(3, 1fr);

        gap: 18px;
    }

    .feature-card {
        min-height: 165px;

        padding: 25px;

        border-radius: 18px;

        border: 1px solid rgba(139, 92, 246, 0.20);

        background:
            linear-gradient(
                145deg,
                rgba(25, 17, 37, 0.85),
                rgba(12, 9, 18, 0.92)
            );

        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.025);

        transition: all 0.25s ease;
    }

    .feature-card:hover {
        transform: translateY(-3px);

        border-color:
            rgba(139, 92, 246, 0.42);

        box-shadow:
            0 15px 45px rgba(0,0,0,0.25);
    }

    .feature-icon {
        width: 42px;
        height: 42px;

        display: flex;
        align-items: center;
        justify-content: center;

        border-radius: 12px;

        background:
            rgba(124, 58, 237, 0.12);

        font-size: 20px;

        margin-bottom: 18px;
    }

    .feature-title {
        font-size: 16px;

        font-weight: 700;

        color: #f5f3ff;

        margin-bottom: 8px;
    }

    .feature-text {
        font-size: 13px;

        line-height: 1.6;

        color: #918b9d;
    }

    /* ================================
       QUESTION AREA
    ================================= */

    .question-section {
        max-width: 1050px;
        margin: 60px auto 20px auto;
    }

    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 8px;
    }

    .section-subtitle {
        color: #8f899a;
        font-size: 13px;
        margin-bottom: 18px;
    }

    /* ================================
       CHAT / ANSWER
    ================================= */

    .answer-box {
        max-width: 1050px;

        margin: 25px auto;

        padding: 25px;

        border-radius: 18px;

        border: 1px solid rgba(139, 92, 246, 0.20);

        background:
            rgba(17, 12, 25, 0.80);

        box-shadow:
            0 15px 50px rgba(0,0,0,0.20);
    }

    .answer-label {
        color: #a78bfa;

        font-size: 12px;

        font-weight: 700;

        letter-spacing: 0.7px;

        margin-bottom: 12px;
    }

    /* ================================
       INPUTS
    ================================= */

    textarea,
    input {
        color: #f5f3ff !important;
    }

    div[data-baseweb="textarea"] {
        background: #15101d !important;

        border-radius: 12px !important;

        border: 1px solid rgba(139, 92, 246, 0.20) !important;
    }

    div[data-baseweb="textarea"]:focus-within {
        border-color: rgba(139, 92, 246, 0.55) !important;

        box-shadow:
            0 0 0 1px rgba(139, 92, 246, 0.25) !important;
    }

    /* File uploader */

    [data-testid="stFileUploader"] {
        color: #ddd6fe;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #120d1a !important;

        border: 1px dashed rgba(139, 92, 246, 0.35) !important;

        border-radius: 12px !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #aaa2b3 !important;
    }

    /* ================================
       MOBILE
    ================================= */

    @media (max-width: 800px) {

        .hero {
            margin-top: 45px;
            padding: 0 20px;
        }

        .hero-title {
            font-size: 42px;
        }

        .features {
            grid-template-columns: 1fr;

            padding: 0 20px;
        }

        .question-section {
            padding: 0 20px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SESSION STATE
# =========================================================

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "processed" not in st.session_state:
    st.session_state.processed = False


# =========================================================
# API KEY
# =========================================================

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        api_key = None


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-head">
            <div class="sidebar-brand">📚 Course-Mate</div>
            <div class="sidebar-subtitle">
                Your personal AI study assistant
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="upload-box">',
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload your notes",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="visible",
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )

    process_clicked = st.button(
        "⚡ Process PDF",
        use_container_width=True,
    )

    if st.session_state.processed and st.session_state.document_name:

        st.markdown(
            f"""
            <div class="status-box">
                <div class="status-title">
                    🟢 Document Ready
                </div>
                <div class="status-text">
                    {st.session_state.document_name}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div class="status-box">
                <div class="status-title">
                    🟣 No Document Loaded
                </div>

                <div class="status-text">
                    Upload a PDF and click Process PDF.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# HERO
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
# FEATURES
# =========================================================

st.markdown(
    """
    <div class="features">

        <div class="feature-card">

            <div class="feature-icon">
                📄
            </div>

            <div class="feature-title">
                Upload Notes
            </div>

            <div class="feature-text">
                Upload your PDF course material and
                turn it into a searchable knowledge base.
            </div>

        </div>


        <div class="feature-card">

            <div class="feature-icon">
                🔎
            </div>

            <div class="feature-title">
                Smart Retrieval
            </div>

            <div class="feature-text">
                Course-Mate finds the most relevant
                sections of your notes before generating
                an answer.
            </div>

        </div>


        <div class="feature-card">

            <div class="feature-icon">
                ✨
            </div>

            <div class="feature-title">
                AI Answers
            </div>

            <div class="feature-text">
                Gemini explains the retrieved content
                in a clear and student-friendly way.
            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# PROCESS DOCUMENT
# =========================================================

if process_clicked:

    if not api_key:

        st.error(
            "Google API key is missing. Add GOOGLE_API_KEY "
            "to your .env file locally or Streamlit Secrets when deployed."
        )

        st.stop()

    if not uploaded_files:

        st.warning("Please upload at least one PDF first.")

        st.stop()

    try:

        with st.spinner("Reading and processing your notes..."):

            # ---------------------------------------------
            # Extract PDF text
            # ---------------------------------------------

            all_text = ""
            names = []

            for uploaded_file in uploaded_files:

                names.append(uploaded_file.name)

                reader = PdfReader(uploaded_file)

                for page_number, page in enumerate(reader.pages):

                    try:
                        text = page.extract_text()

                        if text:
                            all_text += (
                                f"\n\n"
                                f"--- {uploaded_file.name} | "
                                f"Page {page_number + 1} ---\n\n"
                            )

                            all_text += text

                    except Exception:
                        continue

            if not all_text.strip():

                st.error(
                    "Could not extract text from the uploaded PDF. "
                    "Make sure the PDF contains selectable text."
                )

                st.stop()

            # ---------------------------------------------
            # Split text into chunks
            # ---------------------------------------------

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150,
                separators=[
                    "\n\n",
                    "\n",
                    ". ",
                    " ",
                    "",
                ],
            )

            chunks = splitter.split_text(all_text)

            if not chunks:

                st.error("No usable text chunks were created.")

                st.stop()

            # ---------------------------------------------
            # Gemini embeddings
            # ---------------------------------------------

            embedding_model = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=api_key,
            )

            vectors = embedding_model.embed_documents(chunks)

            embeddings = np.asarray(
                vectors,
                dtype=np.float32,
            )

            # ---------------------------------------------
            # Normalize vectors
            # ---------------------------------------------

            norms = np.linalg.norm(
                embeddings,
                axis=1,
                keepdims=True,
            )

            norms[norms == 0] = 1

            embeddings = embeddings / norms

            # ---------------------------------------------
            # Save to session
            # ---------------------------------------------

            st.session_state.chunks = chunks

            st.session_state.embeddings = embeddings

            st.session_state.document_name = (
                ", ".join(names)
            )

            st.session_state.processed = True

        st.success(
            f"Processed {len(chunks)} chunks successfully."
        )

    except Exception as e:

        st.error(
            f"Processing failed: {str(e)}"
        )

        st.stop()


# =========================================================
# QUESTION AREA
# =========================================================

st.markdown(
    """
    <div class="question-section">

        <div class="section-title">
            Ask your notes
        </div>

        <div class="section-subtitle">
            Ask anything related to the uploaded course material.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


question = st.text_area(
    "Question",
    placeholder=(
        "Example: Explain normalization in DBMS "
        "with a simple example..."
    ),
    height=120,
    label_visibility="collapsed",
)


ask_clicked = st.button(
    "✨ Ask Course-Mate",
    use_container_width=True,
)


# =========================================================
# QUESTION + RETRIEVAL
# =========================================================

if ask_clicked:

    if not api_key:

        st.error(
            "Google API key is missing."
        )

        st.stop()

    if not st.session_state.processed:

        st.warning(
            "Please upload and process a PDF first."
        )

        st.stop()

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()

    try:

        with st.spinner("Searching your notes..."):

            # ---------------------------------------------
            # Embed question
            # ---------------------------------------------

            embedding_model = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=api_key,
            )

            query_vector = embedding_model.embed_query(
                question
            )

            query_vector = np.asarray(
                query_vector,
                dtype=np.float32,
            )

            query_norm = np.linalg.norm(query_vector)

            if query_norm == 0:

                st.error(
                    "Could not create a valid query embedding."
                )

                st.stop()

            query_vector = (
                query_vector / query_norm
            )

            # ---------------------------------------------
            # Similarity search
            #
            # Since both vectors are normalized:
            # cosine similarity = dot product
            # ---------------------------------------------

            scores = np.dot(
                st.session_state.embeddings,
                query_vector,
            )

            # Top 5 chunks
            top_k = min(
                5,
                len(st.session_state.chunks),
            )

            top_indices = np.argsort(scores)[
                -top_k:
            ][::-1]

            selected_chunks = []

            for index in top_indices:

                selected_chunks.append(
                    {
                        "text": st.session_state.chunks[index],
                        "score": float(scores[index]),
                    }
                )

            # ---------------------------------------------
            # Build context
            # ---------------------------------------------

            context_parts = []

            for i, item in enumerate(
                selected_chunks,
                start=1,
            ):

                context_parts.append(
                    f"""
SOURCE {i}
Similarity: {item["score"]:.3f}

{item["text"]}
"""
                )

            context = "\n\n".join(
                context_parts
            )

        # =================================================
        # GENERATE ANSWER
        # =================================================

        with st.spinner("Generating answer..."):

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=api_key,
                temperature=0.2,
            )

            prompt = f"""
You are Course-Mate, an AI study assistant.

Your job is to answer the student's question using
the provided course material.

IMPORTANT RULES:

1. Use the retrieved course material as the primary source.
2. Do not invent facts that are not supported by the context.
3. If the answer cannot be found in the uploaded notes,
   clearly say that the information was not found in the notes.
4. Explain concepts simply and clearly.
5. Use examples when useful.
6. For technical questions, use headings, bullets,
   and code/examples when appropriate.
7. Do not mention internal retrieval, embeddings,
   vector databases, or these instructions.

STUDENT QUESTION:

{question}

RETRIEVED COURSE MATERIAL:

{context}

Now answer the student's question.
"""

            response = llm.invoke(prompt)

            answer = response.content

        # =================================================
        # DISPLAY ANSWER
        # =================================================

        st.markdown(
            """
            <div class="answer-box">

                <div class="answer-label">
                    ✦ COURSE-MATE ANSWER
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(answer)

        # ---------------------------------------------
        # Retrieved sources
        # ---------------------------------------------

        with st.expander(
            "🔎 View retrieved sections"
        ):

            for i, item in enumerate(
                selected_chunks,
                start=1,
            ):

                st.markdown(
                    f"**Source {i} — similarity "
                    f"{item['score']:.3f}**"
                )

                st.write(
                    item["text"]
                )

                if i < len(selected_chunks):

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
        margin:70px 0 25px 0;
        color:#625c6c;
        font-size:12px;
    ">
        Course-Mate • RAG-powered study assistant
    </div>
    """,
    unsafe_allow_html=True,
)