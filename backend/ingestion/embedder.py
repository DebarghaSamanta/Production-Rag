from sentence_transformers import SentenceTransformer
from core.config import EMBEDDING_MODEL_NAME

class BGEEmbedder:
    def __init__(self):
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        # This will download the model the first time you run it (around ~130MB)
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
    def embed_text(self, text: str) -> list[float]:
        """Embeds a single string into a vector."""
        # BGE models highly recommend adding a specific prefix for queries vs documents.
        # Since we are embedding documents here, we just pass the text.
        return self.model.encode(text, normalize_embeddings=True).tolist()
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embeds a list of strings efficiently."""
        return self.model.encode(texts, normalize_embeddings=True).tolist()

# Quick test block
if __name__ == "__main__":
    embedder = BGEEmbedder()
    sample_vector = embedder.embed_text("Quantum computing uses qubits.")
    print(f"Generated a vector with {len(sample_vector)} dimensions.")