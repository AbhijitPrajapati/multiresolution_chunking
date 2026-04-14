from chromadb import Collection
from src.embedding import embedding_model


def retrieve_chunks(prompts, n_chunks, collection: Collection):
    return collection.query(
        query_embeddings=embedding_model.encode(prompts),
        n_results=n_chunks,
    )["documents"]
