from nltk import sent_tokenize
from src.chunking.util import chunk_sentences_semantically


def semantic_chunking(sections, threshold):
    chunks = []
    section_names = []
    for section in sections:
        c = chunk_sentences_semantically(
            sent_tokenize(section["text"]),
            threshold,
            1,
        )
        chunks.extend(c)
        section_names.extend([section["section"]] * len(c))
    return chunks, section_names
