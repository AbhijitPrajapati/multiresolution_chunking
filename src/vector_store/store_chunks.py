from src.nlp import embedding_model
from uuid import uuid4
from .client import client


def store_chunks(method, fn, title, chunks, section_names):
    embeddings = embedding_model.encode(chunks)
    ids = [f"{fn}_{str(uuid4())}" for _ in range(len(chunks))]
    mds = [{"title": title, "section": s} for s in section_names]
    collection = client.get_collection(method)
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=mds,  # type: ignore
    )
