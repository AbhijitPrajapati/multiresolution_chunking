from nltk.tokenize import word_tokenize
from src.constants import FIXED_TOKENS


def fixed_length_chunking(sections):
    chunks = []
    section_names = []
    for section in sections:
        i = 0
        words = word_tokenize(section["text"])
        while i < len(words):
            chunk_words = words[i : i + FIXED_TOKENS]
            chunks.append(" ".join(chunk_words))
            section_names.append(section["section"])
            i += int(0.8 * FIXED_TOKENS)
    return chunks, section_names
