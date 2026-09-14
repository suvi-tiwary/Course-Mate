from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import MistralAIEmbeddings 
from langchain_chroma import Chroma

load_dotenv()


embedding_model = MistralAIEmbeddings(
     model="mistral-embed"
)

vector_store = Chroma(
    embedding_function=embedding_model,
    persist_directory="chromaDB",
)

if vector_store._collection.count() == 0:
    raise RuntimeError(
        "The Chroma database is empty. Run `python create_db.py` before starting the app."
    )

retriver = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":2,                   ## then apply the mmr to get 2
        "fetch_k":8,             ## give 8 ans by similarity search
        "lambda_mult":0.5        ## show the diversity
    }
)

llm = ChatGoogleGenerativeAI(model="gemini-3.7-flash")

prompt = ChatPromptTemplate.from_messages([
    ("system","""you are a helpful ai assitant .
    use ONLY the provided context to answer the questions.
    if the answer is not present in the context ,
    say: i could not find the answer in the context  
    """),

    ('user',"""Context:
    {context}
    Question:
    {question} 
    """)
])

print("############## RAG PROJECT ##############")
print("Press 0 to exit ")

while True:
    query= input("you : ")
    if query=="0":
        print("Bye Bye peyareee")
        break
    docs = retriver.invoke(query)
    context = "\n\n".join(
        doc.page_content for doc in docs
    )


    final_prompt = prompt.invoke(
       { "context":context,
        "question":query}
    )

    try:
        response = llm.invoke(final_prompt)
    except Exception as error:
        print(f"\nAI request failed: {error}")
        print("Check your GEMINI_API_KEY and network connection, then try again.")
        continue

    print("\n\n AI : ",response.text)





