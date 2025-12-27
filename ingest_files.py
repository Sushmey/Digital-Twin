from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

def ingest_text_to_pinecone(
    text: str,
    index: str,
    filename: str,
    namespace: str
):
    # 1. Chunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    docs = text_splitter.create_documents([text])

    # 2. Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL")
    )

    # 3. Pinecone init
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pc.Index(index)

    # 3. Add metadata to each doc
    for doc in docs:
        if doc.metadata is None:
            doc.metadata = {}
        doc.metadata["filename"] = filename

    # 4. Store in Pinecone
    PineconeVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=index,
        namespace=namespace
    )

    return len(docs)
