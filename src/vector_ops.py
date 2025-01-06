"""
Vector operations for text similarity and embedding.
"""
import math
from collections import Counter

def create_word_vector(text, vocabulary=None):
    """Create a simple bag-of-words vector."""
    words = text.lower().split()
    if vocabulary is None:
        vocabulary = sorted(set(words))
    
    vector = [words.count(word) for word in vocabulary]
    return vector, vocabulary

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_product / (norm1 * norm2)

def find_similar_questions(question, question_vectors, threshold=0.8):
    """Find similar questions using cosine similarity."""
    target_vector, _ = create_word_vector(question, question_vectors['vocabulary'])
    similar = []
    
    for idx, vec in enumerate(question_vectors['vectors']):
        similarity = cosine_similarity(target_vector, vec)
        if similarity > threshold:
            similar.append((idx, similarity))
    
    return sorted(similar, key=lambda x: x[1], reverse=True)