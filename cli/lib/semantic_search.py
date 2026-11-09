import os
import json
import numpy as np
import re
from sentence_transformers import SentenceTransformer


class SemanticSearch:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        if not text.strip():
            raise ValueError("Input text cannot be empty or only whitespace")

        result = self.model.encode([text])
        return result[0]

    def build_embeddings(self, documents):
        self.documents = documents

        self.document_map = {
            document["id"]: document
            for document in self.documents
        }

        doc_list = [
            f"{doc['title']}: {doc['description']}"
            for doc in self.documents
        ]

        self.embeddings = self.model.encode(
            doc_list,
            show_progress_bar=True
        )

        np.save("cache/movie_embeddings.npy", self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents

        self.document_map = {
            document["id"]: document
            for document in self.documents
        }

        if os.path.exists("cache/movie_embeddings.npy"):
            self.embeddings = np.load("cache/movie_embeddings.npy")

            if len(self.embeddings) == len(self.documents):
                return self.embeddings

        return self.build_embeddings(documents)
    
    def search(self, query, limit):
        
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        
        text_embeddings = self.generate_embedding(query)
        
        similarities = [
            cosine_similarity(embedding, text_embeddings)
            for embedding in self.embeddings
        ]
        
        score_list = [
            (score, document) for score, document in zip(similarities, self.documents) ]
        
        sorted_results = sorted(
            score_list,
            key=lambda x: x[0],
            reverse=True
        )
        
        top_results = sorted_results[:limit]
        
        return [
                {
                    "score": score,
                    "title": document["title"],
                    "description": document["description"]
                }
                for score, document in top_results
        ]


def embed_text(text: str):
    engine = SemanticSearch()
    embedding = engine.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

    return embedding


def verify_model():
    engine = SemanticSearch()

    print(f"Model loaded: {engine.model}")
    print(f"Max sequence length: {engine.model.max_seq_length}")


def verify_embeddings():
    engine = SemanticSearch()

    with open("data/movies.json", "r", encoding="utf-8") as f:
        movie_data = json.load(f)
        movie_list = movie_data["movies"]

    print(type(movie_list))
    print(type(movie_list[0]) if isinstance(movie_list, list) else movie_list.keys())

    embeddings = engine.load_or_create_embeddings(movie_list)

    print(f"Number of docs:   {len(engine.documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors "
        f"in {embeddings.shape[1]} dimensions"
    )

def embed_query_text(query):
    engine = SemanticSearch()
    embedding = engine.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")
    

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def semantic_search(query, limit):
    engine = SemanticSearch()

    with open("data/movies.json", "r", encoding="utf-8") as f:
        movie_data = json.load(f)
        movie_list = movie_data["movies"]

    engine.load_or_create_embeddings(movie_list)

    results = engine.search(query, limit)

    max_chars = 200

    for i, result in enumerate(results, start=1):
        description = result["description"]

        if len(description) > max_chars:
            description = description[:max_chars] + "..."

        print(f"{i}. {result['title']}: (score: {result['score']:.4f})")
        print(f"  {description}")
        print()
        
def chunk_text(text, chunk_size, overlap):
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be non-negative and smaller than chunk_size")

    words = text.strip().split()

    if not words:
        return []

    step = chunk_size - overlap

    chunks = [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), step)
    ]

    print(f"Chunking {len(text)} characters")

    for i, chunk in enumerate(chunks, start=1):
        print(f"{i}. {chunk}")

    return chunks

def semantic_chunk(text, max_chunk_size=4, overlap=1):
    if max_chunk_size <= 0:
        raise ValueError("Max chunk size must be positive")

    if overlap < 0 or overlap >= max_chunk_size:
        raise ValueError("Invalid overlap")

    text = text.strip()

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)

    # Handle text consisting of one unpunctuated sentence
    if len(sentences) == 1 and not text.endswith((".", "!", "?")):
        sentences = [text]

    chunks = []
    i = 0

    while i < len(sentences):
        chunk_sentences = sentences[i:i + max_chunk_size]

        chunk_sentences = [
            sentence.strip()
            for sentence in chunk_sentences
        ]

        chunk_sentences = [
            sentence
            for sentence in chunk_sentences
            if sentence
        ]

        # Avoid chunks containing only overlap
        if chunks and len(chunk_sentences) <= overlap:
            break

        if chunk_sentences:
            chunks.append(" ".join(chunk_sentences))

        i += max_chunk_size - overlap

    return chunks

def semantic_chunk_text(text, max_chunk_size=4, overlap=0):
    chunks = semantic_chunk(text, max_chunk_size, overlap)

    print(f"Semantically chunking {len(text)} characters")

    for i, chunk in enumerate(chunks, start=1):
        print(f"{i}. {chunk}")

    return chunks