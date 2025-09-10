from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from sentence_transformers import SentenceTransformer


# Create a new class to load the sentence transformer model on CPU
class HuggingFaceEmbeddingsCpu(HuggingFaceEmbeddings):
    def __init__(self, model_name: str):
        # Call the original HuggingFaceEmbeddings initialization
        super().__init__(model_name=model_name)
        # Load the sentence-transformers model explicitly on the CPU
        self.client = SentenceTransformer(model_name, device="cpu")


# This function creates a FAISS index from a list of text chunks
def create_faiss_index(texts: List[str]):
    # Create embeddings using the CPU-only class above
    embeddings = HuggingFaceEmbeddingsCpu(model_name="sentence-transformers/all-mpnet-base-v2")
    # Create and return the FAISS index from the texts and embeddings
    return FAISS.from_texts(texts, embeddings)


# This function searches for documents similar to the query in the FAISS index
def retrive_relevant_docs(vectorstore: FAISS, query: str, k: int = 3):
    # Return the top 'k' most similar documents to the query
    return vectorstore.similarity_search(query, k=k)
