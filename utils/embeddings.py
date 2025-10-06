from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding_model():
    return model

def embed_text(texts):
    """
    Generate embeddings for a list of texts.
    Returns list of 384-dim vectors.
    """
    embeddings = model.encode(texts)
    return embeddings.tolist()  # Convert to list for Supabase