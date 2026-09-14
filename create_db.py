from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings 
from langchain_chroma import Chroma

load_dotenv()


## data loading
data = TextLoader("documents/DBMS.txt")
docs=data.load()
# print(docs[0].page_content)


spilter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=70
)
chunks = spilter.split_documents(docs)
print(len(chunks))


embedding_model = MistralAIEmbeddings(
    model='mistral-embed',    
)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chromaDB"
)