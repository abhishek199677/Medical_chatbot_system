from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from sentence_transformers import SentenceTransformer

class HuggingFaceEmbeddingsCpu(HuggingFaceEmbeddings):
    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        # Force model to CPU
        self.client = SentenceTransformer(model_name, device="cpu")


def create_faiss_index(texts: List[str]):
    embeddings = HuggingFaceEmbeddingsCpu(model_name="sentence-transformers/all-mpnet-base-v2")
    return FAISS.from_texts(texts, embeddings)


def retrive_relevant_docs(vectorstore: FAISS, query: str, k: int = 3):
    return vectorstore.similarity_search(query, k=k)
