from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai import configure
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import os
from pinecone import Pinecone


load_dotenv()


configure(api_key=os.getenv("GEMINI_API_KEY"))
def answer_user_question(
    question: str,
    index: str,
    namespace: str
):
    # 1. Embeddings (same model as ingestion)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    # 2. Connect to Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pc.Index(index)
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index,
        embedding=embeddings,
        namespace=namespace
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 8}
    )

    # 3. LLM
    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-2.0-flash",
    #     temperature=0.2
    # )

    llm = ChatCohere(
        model="command-a-03-2025",
        temperature=0.1,
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        max_output_tokens = 300,
    )

    # 4. RAG chain
    system_template = os.getenv("SYSTEM_TEMPLATE")

    human_template = "{question}\n\nContext:\n{context}"

    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": chat_prompt}
    )

    result = qa(question)
    print(result)
    return {
        "answer": result["result"],
        "sources": [
            result["source_documents"][0].metadata.get("filename")
        ]
    }
