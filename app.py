
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# CONFIG
# =========================================================

load_dotenv()

st.set_page_config(
    page_title="RAG Book Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG Book Assistant")
st.write("Upload a PDF and ask questions from the document.")


# =========================================================
# CHECK GEMINI API KEY
# =========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    try:
        GEMINI_API_KEY = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        GEMINI_API_KEY = None


# =========================================================
# SESSION STATE
# =========================================================

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None


# =========================================================
# PDF UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📄 Upload a PDF book",
    type=["pdf"]
)


if uploaded_file:

    st.success(f"Uploaded: {uploaded_file.name}")

    # =====================================================
    # CREATE VECTOR DATABASE
    # =====================================================

    if st.button(
        "🚀 Create Vector Database",
        use_container_width=True
    ):

        try:

            with st.spinner("Reading PDF..."):

                # Create temporary PDF
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as tmp_file:

                    tmp_file.write(
                        uploaded_file.getvalue()
                    )

                    file_path = tmp_file.name


                # Load PDF
                loader = PyPDFLoader(file_path)

                documents = loader.load()


            with st.spinner(
                "Splitting document into chunks..."
            ):

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                chunks = splitter.split_documents(
                    documents
                )


            st.info(
                f"📄 Pages: {len(documents)} | "
                f"🧩 Chunks: {len(chunks)}"
            )


            # =================================================
            # HUGGING FACE EMBEDDINGS
            # =================================================

            with st.spinner(
                "Creating embeddings..."
            ):

                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={
                        "device": "cpu"
                    },
                    encode_kwargs={
                        "normalize_embeddings": True
                    }
                )


            # =================================================
            # IN-MEMORY CHROMA
            # =================================================

            with st.spinner(
                "Creating vector database..."
            ):

                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings
                )


            # Store vector database in session
            st.session_state.vectorstore = vectorstore

            st.session_state.file_name = (
                uploaded_file.name
            )


            # Delete temporary PDF
            try:
                os.remove(file_path)
            except Exception:
                pass


            st.success(
                "✅ Vector database created successfully!"
            )


        except Exception as e:

            st.error(
                "❌ Processing failed."
            )

            with st.expander(
                "Show processing error"
            ):
                st.code(str(e))


# =========================================================
# QUESTION ANSWERING
# =========================================================

if st.session_state.vectorstore is not None:

    st.divider()

    st.subheader(
        "📖 Ask Questions From the Book"
    )

    st.caption(
        f"Currently loaded: "
        f"{st.session_state.file_name}"
    )


    # =====================================================
    # RETRIEVER
    # =====================================================

    retriever = (
        st.session_state.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )
    )


    # =====================================================
    # GEMINI API KEY CHECK
    # =====================================================

    if not GEMINI_API_KEY:

        st.warning(
            "⚠️ GOOGLE_API_KEY is missing."
        )

        st.info(
            "Add GEMINI_API_KEY in "
            "Streamlit Cloud → "
            "Manage app → Settings → Secrets."
        )


    else:

        # =================================================
        # GEMINI LLM
        # =================================================

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0
        )


        # =================================================
        # PROMPT
        # =================================================

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are a helpful AI assistant for answering
questions from a user's uploaded book or PDF.

Use ONLY the information provided in the context.

Do not use outside knowledge.

If the answer cannot be found in the context,
say exactly:

"I could not find the answer in the document."

Keep the answer clear and easy to understand.
"""
                ),

                (
                    "human",
                    """
Context:

{context}


Question:

{question}
"""
                )
            ]
        )


        # =================================================
        # USER QUESTION
        # =================================================

        query = st.text_input(
            "💬 Enter your question",
            placeholder=(
                "Example: What is normalization in DBMS?"
            )
        )


        if query:

            try:

                # =========================================
                # RETRIEVE DOCUMENTS
                # =========================================

                with st.spinner(
                    "🔎 Searching the document..."
                ):

                    docs = retriever.invoke(query)


                if not docs:

                    st.warning(
                        "No relevant information was "
                        "found in the document."
                    )


                else:

                    # =====================================
                    # CREATE CONTEXT
                    # =====================================

                    context = "\n\n".join(
                        [
                            doc.page_content
                            for doc in docs
                        ]
                    )


                    # =====================================
                    # CREATE PROMPT
                    # =====================================

                    final_prompt = prompt.invoke(
                        {
                            "context": context,
                            "question": query
                        }
                    )


                    # =====================================
                    # CALL GEMINI
                    # =====================================

                    with st.spinner(
                        "🤖 Generating answer..."
                    ):

                        response = llm.invoke(
                            final_prompt
                        )


                    # =====================================
                    # DISPLAY ANSWER
                    # =====================================

                    st.write(
                        "### 🤖 AI Answer"
                    )

                    st.write(
                        response.content
                    )


                    # =====================================
                    # SHOW SOURCES
                    # =====================================

                    with st.expander(
                        "📚 View Retrieved Sources"
                    ):

                        for i, doc in enumerate(docs):

                            page_number = (
                                doc.metadata.get(
                                    "page",
                                    "Unknown"
                                )
                            )

                            if isinstance(
                                page_number,
                                int
                            ):
                                display_page = (
                                    page_number + 1
                                )
                            else:
                                display_page = (
                                    page_number
                                )


                            st.markdown(
                                f"**Source {i + 1} "
                                f"(Page {display_page})**"
                            )

                            st.write(
                                doc.page_content[:1000]
                            )

                            st.divider()


            except Exception as e:

                st.error(
                    "❌ Gemini API request failed."
                )

                with st.expander(
                    "🔧 Show Gemini error details"
                ):

                    st.code(str(e))


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "📚 Course-Mate RAG • "
    "Hugging Face Embeddings + "
    "Chroma + Gemini"
)
