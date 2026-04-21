from chromadb import PersistentClient
from src.nlp import embedding_model
from .client import client


def retrieve_chunks(prompts, n_chunks, method):
    collection = client.get_collection(method)
    return collection.query(
        query_embeddings=embedding_model.encode(prompts),
        n_results=n_chunks,
    )["documents"]
