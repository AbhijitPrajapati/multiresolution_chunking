from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5", backend="torch")


def chunk_sentences_semantically(sentences, similarity_threshold, sentence_overlap):
    embeddings = embedding_model.encode(sentences)
    similarities = embedding_model.similarity_pairwise(embeddings[:-1], embeddings[1:])

    dip_idx = [
        i for i in range(len(similarities)) if similarities[i] < similarity_threshold
    ]

    chunks = []
    current = [sentences[0]]
    for i, s in enumerate(sentences[1:]):
        if i in dip_idx:
            chunks.append(" ".join(current))
            ovr = min(sentence_overlap, len(current))
            current = current[-ovr:] + [sentences[i + 1]]
        else:
            current.append(s)
    if current:
        chunks.append(" ".join(current))
    return chunks
