import streamlit as st
from dotenv import load_dotenv
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
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
st.write("Upload a PDF and ask questions from the document")


# =========================================================
# PDF UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload a PDF book",
    type="pdf"
)


if uploaded_file:

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("PDF uploaded successfully!")


    # =====================================================
    # CREATE VECTOR DATABASE
    # =====================================================

    if st.button("Create Vector Database"):

        with st.spinner("Processing document..."):

            # Load PDF
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            # Split document
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(docs)

            st.info(f"Created {len(chunks)} document chunks.")


            # =================================================
            # HUGGING FACE EMBEDDINGS
            # =================================================

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )


            # =================================================
            # CREATE CHROMA DATABASE
            # =================================================

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory="chroma_db"
            )

        st.success("Vector database created successfully! 🎉")


# =========================================================
# LOAD VECTOR DATABASE
# =========================================================

if os.path.exists("chroma_db"):

    # Same embedding model must be used
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )


    # =====================================================
    # RETRIEVER
    # =====================================================

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )


    # =====================================================
    # MISTRAL LLM
    # =====================================================

    llm = ChatMistralAI(
        model="mistral-small-2506"
    )


    # =====================================================
    # PROMPT
    # =====================================================

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say exactly:

"I could not find the answer in the document."

Do not use outside knowledge."""
            ),

            (
                "human",
                """Context:

{context}

Question:

{question}"""
            )
        ]
    )


    # =====================================================
    # QUESTION ANSWERING
    # =====================================================

    st.divider()

    st.subheader("📖 Ask Questions From the Book")

    query = st.text_input(
        "Enter your question"
    )


    if query:

        with st.spinner("Searching the document..."):

            # Retrieve relevant chunks
            docs = retriever.invoke(query)

            # Create context
            context = "\n\n".join(
                [
                    doc.page_content
                    for doc in docs
                ]
            )


            # Create final prompt
            final_prompt = prompt.invoke(
                {
                    "context": context,
                    "question": query
                }
            )


            # Ask Mistral
            response = llm.invoke(
                final_prompt
            )


        st.write("### 🤖 AI Answer")

        st.write(response.content)