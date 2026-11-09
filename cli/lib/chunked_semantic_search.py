from .semantic_search import (
    SemanticSearch, semantic_chunk_text, semantic_chunk, cosine_similarity)
import os
import json
import numpy as np



class ChunkedSemanticSearch(SemanticSearch):

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:

        self.documents = documents

        self.document_map = {
            document["id"]: document
            for document in self.documents
        }

        all_chunks = []
        chunk_metadata = []

        for movie_idx, document in enumerate(self.documents):

            description = document["description"]

            if not description or not description.strip():
                continue

            chunks = semantic_chunk(
                description,
                max_chunk_size=4,
                overlap=1
            )

            total_chunks = len(chunks)

            for chunk_idx, chunk in enumerate(chunks):

                all_chunks.append(chunk)

                chunk_metadata.append({
                    "movie_idx": movie_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": total_chunks
                })

        self.chunk_embeddings = self.model.encode(
            all_chunks,
            show_progress_bar=True
        )

        self.chunk_metadata = chunk_metadata

        os.makedirs("cache", exist_ok=True)

        np.save(
            "cache/chunk_embeddings.npy",
            self.chunk_embeddings
        )

        with open("cache/chunk_metadata.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "chunks": chunk_metadata,
                    "total_chunks": len(all_chunks)
                },
                f,
                indent=2
            )

        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(
        self,
        documents: list[dict]
    ) -> np.ndarray:

        self.documents = documents

        self.document_map = {
            document["id"]: document
            for document in self.documents
        }

        embeddings_path = "cache/chunk_embeddings.npy"
        metadata_path = "cache/chunk_metadata.json"

        if (
            os.path.exists(embeddings_path)
            and os.path.exists(metadata_path)
        ):

            self.chunk_embeddings = np.load(embeddings_path)

            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata_data = json.load(f)

            self.chunk_metadata = metadata_data["chunks"]

            return self.chunk_embeddings
        
        return self.build_chunk_embeddings(documents)
    
    def search_chunks(self, query: str, limit: int = 10):
        query_embedding = self.generate_embedding(query)
        chunk_scores = []
        
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            score = cosine_similarity(chunk_embedding, query_embedding)
            
            metadata = self.chunk_metadata[i]
            
            chunk_scores.append({
                "chunk_idx" : metadata["chunk_idx"],
                "movie_idx" : metadata["movie_idx"],
                "score" : score
            })
        
        movie_scores = {}
        
        for chunk in chunk_scores:
            movie_idx = chunk["movie_idx"]
            score = chunk["score"]
            
            if movie_idx not in movie_scores or score > movie_scores[movie_idx]:
                movie_scores[movie_idx] = score
        
        sorted_movies = sorted(
            movie_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )
        
        top_movies = sorted_movies[:limit]
        
        results = []
        
        for movie_idx, score in top_movies:
            doc = self.documents[movie_idx]
            
            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "document": doc["description"][:100],
                "score": round(float(score),3)
            })
            
        return results