"""
Embedding functionality using Ollama's embeddinggemma model.
"""

import ollama
import time
from typing import List, Optional


def create_embedding(
    text: str, model_name: str = "embeddinggemma"
) -> Optional[List[float]]:
    """
    Create an embedding vector for the given text using embeddinggemma.

    Args:
        text: Text to embed
        model_name: Ollama embedding model to use (default: embeddinggemma)

    Returns:
        List of floats representing the embedding vector, or None if failed
    """
    try:
        start_time = time.time()

        # Use Ollama's embeddings endpoint
        response = ollama.embeddings(model=model_name, prompt=text)

        embedding_time = time.time() - start_time
        embedding = response["embedding"]

        print(
            f"✅ Created embedding ({len(embedding)} dimensions) in {round(embedding_time, 3)}s"
        )

        return embedding

    except Exception as e:
        print(f"❌ Error creating embedding: {e}")
        return None


def create_embeddings_batch(
    texts: List[str], model_name: str = "embeddinggemma"
) -> List[Optional[List[float]]]:
    """
    Create embeddings for multiple texts.

    Args:
        texts: List of texts to embed
        model_name: Ollama embedding model to use

    Returns:
        List of embedding vectors (or None for failed embeddings)
    """
    embeddings = []

    print(f"🔄 Creating {len(texts)} embeddings...")

    for i, text in enumerate(texts):
        print(f"  [{i+1}/{len(texts)}] Processing...")
        embedding = create_embedding(text, model_name)
        embeddings.append(embedding)

    successful = sum(1 for e in embeddings if e is not None)
    print(f"✅ Created {successful}/{len(texts)} embeddings successfully")

    return embeddings


def warmup_embedding_model(model_name: str = "embeddinggemma") -> bool:
    """
    Warm up the embedding model by running a test embedding.

    Args:
        model_name: Ollama embedding model to warm up

    Returns:
        True if warmup successful, False otherwise
    """
    try:
        print(f"🔥 Warming up embedding model ({model_name})...")
        start_time = time.time()

        # Create a test embedding
        test_text = "Test embedding"
        response = ollama.embeddings(model=model_name, prompt=test_text)

        warmup_time = time.time() - start_time
        embedding_dim = len(response["embedding"])

        print(f"✅ Embedding model warmed up in {round(warmup_time, 2)}s")
        print(f"   Embedding dimensions: {embedding_dim}")

        return True

    except Exception as e:
        print(f"⚠️  Embedding model warmup failed: {e}")
        print("   Make sure to pull the model: ollama pull embeddinggemma")
        return False
