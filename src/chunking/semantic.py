from nltk import sent_tokenize
from src.embedding import embedding_model
from src.constants import SIMILARITY_THRESHOLD


def semantic_chunking(sections):
    chunks = []
    section_names = []
    for section in sections:
        sentences = sent_tokenize(section["text"])
        embeddings = embedding_model.encode(sentences)
        similarities = embedding_model.similarity_pairwise(
            embeddings[:-1], embeddings[1:]
        )

        dip_idx = [
            i
            for i in range(len(similarities))
            if similarities[i] < SIMILARITY_THRESHOLD
        ]

        c = []
        current = [sentences[0]]
        for i, s in enumerate(sentences[1:]):
            if i in dip_idx:
                c.append(" ".join(current))
                ovr = min(1, len(current))
                current = current[-ovr:] + [sentences[i + 1]]
            else:
                current.append(s)
        if current:
            c.append(" ".join(current))
        chunks.extend(c)
        section_names.extend([section["section"]] * len(c))
    return chunks, section_names
