from chromadb import Collection
from src.embedding import embedding_model


def retrieve_chunks(prompts, n_chunks, collection: Collection, md_filter=None):
    return collection.query(
        query_embeddings=embedding_model.encode(prompts),
        where=md_filter,  # type: ignore
        n_results=n_chunks,
    )["documents"]
