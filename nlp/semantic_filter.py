from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

def is_redundant(text, existing_embeddings, threshold=0.85):
    new_embedding = model.encode([text])
    similarities = cosine_similarity(new_embedding, existing_embeddings)
    return similarities.max() > threshold