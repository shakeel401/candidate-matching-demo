import os
from dotenv import load_dotenv
from uuid import uuid4
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Configs
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME", "candidate-index")

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)

# Create index if it doesn't exist
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Ensure region matches your Pinecone project
    )

# Load the index
index = pc.Index(index_name)

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize LangChain vector store
vector_store = PineconeVectorStore(index=index, embedding=embedding_model)

def insert_documents(resume_docs: list[Document]):
    """
    Inserts single-chunk, GPT-refined resume documents into Pinecone.
    
    Args:
        resume_docs: List of Document objects with full resume content and metadata.
    """
    ids = [str(uuid4()) for _ in range(len(resume_docs))]
    vector_store.add_documents(resume_docs, ids=ids)

def search_similar_docs(query: str, k: int = 2):
    """
    Searches Pinecone for top-k resumes matching the job description query.
    
    Args:
        query: Job description or recruiter query.
        k: Number of top matches to retrieve.

    Returns:
        List[dict]: Each dict contains:
            - resume_id
            - confidence_score (1 - distance)
            - resume_text (preprocessed)
            - metadata
    """
    results = vector_store.similarity_search_with_score(query, k=k)

    if not results:
        return []

    matches = []
    for doc, score in results:
        confidence = round((1 - score) * 100, 2)
        matches.append({
            "resume_id": doc.metadata.get("resume_id"),
            "confidence_score": confidence,
            "resume_text": doc.page_content,
            "metadata": doc.metadata
        })

    return matches
