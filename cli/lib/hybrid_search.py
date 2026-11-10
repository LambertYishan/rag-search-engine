import os

from .keyword_search import InvertedIndex, load_movies
from .chunked_semantic_search import ChunkedSemanticSearch
from sentence_transformers import CrossEncoder

import os
from dotenv import load_dotenv
from openai import OpenAI
import re, time
import json


def correct_spelling(query: str) -> str:
    load_dotenv()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    prompt = f"""Fix any spelling errors in the user-provided movie search query below.
    Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
    Preserve punctuation and capitalization unless a change is required for a typo fix.
    If there are no spelling errors, or if you're unsure, output the original query unchanged.
    Output only the final query text, nothing else.
    User query: "{query}"
    """

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[{"role": "user", "content": prompt}],
    )

    corrected_query = response.choices[0].message.content

    if not corrected_query:
        return query

    return corrected_query.strip()

def rewrite(query: str) -> str:
    load_dotenv()
    
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

    Consider:
    - Common movie knowledge (famous actors, popular films)
    - Genre conventions (horror = scary, animation = cartoon)
    - Keep the rewritten query concise (under 10 words)
    - It should be a Google-style search query, specific enough to yield relevant results
    - Don't use boolean logic

    Examples:
    - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
    - "movie about bear in london with marmalade" -> "Paddington London marmalade"
    - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

    If you cannot improve the query, output the original unchanged.
    Output only the rewritten query text, nothing else.

    User query: "{query}"
    """

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[{"role": "user", "content": prompt}],
    )

    corrected_query = response.choices[0].message.content

    if not corrected_query:
        return query

    return corrected_query.strip()

def expand(query: str) -> str:
    load_dotenv()
    
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    prompt = f"""Expand the user-provided movie search query below with related terms.

    Add synonyms and related concepts that might appear in movie descriptions.
    Keep expansions relevant and focused.
    Output only the additional terms; they will be appended to the original query.

    Examples:
    - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
    - "action movie with bear" -> "action thriller bear chase fight adventure"
    - "comedy with bear" -> "comedy funny bear humor lighthearted"

    User query: "{query}"
    """

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[{"role": "user", "content": prompt}],
    )

    corrected_query = response.choices[0].message.content

    if not corrected_query:
        return query

    return corrected_query.strip()
def rerank(results: list[dict], query: str, method: str) -> list[dict]:

    load_dotenv()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    if method == "individual":

        for i, result in enumerate(results):
            doc = result["document"]

            prompt = f"""Rate how well this movie matches the search query.

            Query: "{query}"
            Movie: {doc.get("title", "")} - {doc.get("description", "")}

            Consider:
            - Direct relevance to query
            - User intent (what they're looking for)
            - Content appropriateness

            Rate 0-10 (10 = perfect match).
            Output ONLY the number in your response, no other text or explanation.

            Score:"""

            response = client.chat.completions.create(
                model="openrouter/free",
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = response.choices[0].message.content

            if not response_text:
                raise ValueError("Empty LLM response")

            score_text = response_text.strip()

            if not re.fullmatch(
                r"(?:10(?:\.0+)?|[0-9](?:\.\d+)?)",
                score_text
            ):
                raise ValueError(f"Invalid rerank score: {response_text}")

            result["rerank_score"] = float(score_text)

            if i < len(results) - 1:
                time.sleep(3)

        return sorted(
            results,
            key=lambda result: result["rerank_score"],
            reverse=True
        )

    elif method == "batch":

        doc_list = []

        for result in results:
            doc = result["document"]

            doc_list.append(
                f"ID: {doc['id']}\n"
                f"Title: {doc['title']}\n"
                f"Description: {doc['description']}"
            )

        doc_list_str = "\n\n".join(doc_list)

        prompt = f"""Rank ALL movies listed below by relevance to the following search query.

        Query: "{query}"

        Movies:
        {doc_list_str}

        IMPORTANT:
        - Return EVERY movie ID listed above.
        - Each movie ID must appear exactly once.
        - Do not invent or omit any IDs.
        - Return exactly {len(results)} movie IDs.
        - Rank all movies from most relevant to least relevant.

        Your response must be a raw JSON array of integers.
        Do not wrap the JSON in Markdown.
        Do not include any explanatory text.

        For example:
        [75, 12, 34, 2, 1]

        Ranking:"""

        response = client.chat.completions.create(
            model="openrouter/free",
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response.choices[0].message.content

        if not response_text:
            raise ValueError("Empty LLM response")

        ranked_ids = json.loads(response_text.strip())
        
        candidate_ids = {
            result["document"]["id"] for result in results
        }

        print("Raw response:", repr(response_text))
        print("Expected IDs:", candidate_ids)
        print("Returned IDs:", ranked_ids)
        print("Missing:", candidate_ids - set(ranked_ids))
        print("Unexpected:", set(ranked_ids) - candidate_ids)
        print("Expected count:", len(candidate_ids))
        print("Returned count:", len(ranked_ids))

        # Validate the returned ranking
        if not isinstance(ranked_ids, list):
            raise ValueError("LLM response must be a JSON list")

        candidate_ids = {
            result["document"]["id"] for result in results
        }

        if (
            len(ranked_ids) != len(candidate_ids)
            or set(ranked_ids) != candidate_ids
            or any(type(movie_id) is not int for movie_id in ranked_ids)
        ):
            raise ValueError(
                "LLM returned missing, duplicate, or invalid movie IDs"
            )

        # Map movie IDs to their original result dictionaries
        result_map = {
            result["document"]["id"]: result
            for result in results
        }

        # Preserve original metadata while changing the order
        return [result_map[movie_id] for movie_id in ranked_ids]

    elif method == "cross_encoder":
        pairs = []

        for result in results:
            doc = result["document"]
            pairs.append([
                query,
                f"{doc.get('title', '')} - {doc.get('description', '')}"
            ])
        
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
        scores = cross_encoder.predict(pairs)
        
        for result, score in zip(results, scores):
            result["cross_encoder_score"] = float(score)
        
        return sorted(
            results,
            key=lambda result: result["cross_encoder_score"],
            reverse=True
        )
        
        
    
    else:
        raise ValueError(f"Unsupported reranking method: {method}")

def normalize(scores:list) -> list:
    if not scores:
        return []

    max_score = max(scores)
    min_score = min(scores)
    denom = max_score - min_score

    if denom == 0:
        normalized = [1.0 for _ in scores]
    else:
        normalized = [
            (score - min_score) / denom
            for score in scores
        ]
    return normalized

def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)

class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        if not 0 <= alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1")
        
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)
        
        bm25_scores = normalize([
            score for doc_id, score in bm25_results
        ])
        
        semantic_scores = normalize([
            result["score"] for result in semantic_results
        ])
        
        combined_results = {}
        
        for (doc_id, score), normalized_score in zip(bm25_results, bm25_scores):
            combined_results[doc_id] = {
                "document": self.idx.docmap[doc_id],
                "bm25_score": normalized_score,
                "semantic_score": 0.0
            }
            
        for result, normalized_score in zip(semantic_results, semantic_scores):
            doc_id = result["id"]
            
            if doc_id not in combined_results:
                combined_results[doc_id] = {
                    "document": self.documents[
                        next(
                            i for i, doc in enumerate(self.documents)
                            if doc["id"] == doc_id
                        )
                    ],
                    "bm25_score": 0.0,
                    "semantic_score": normalized_score
                }
            else:
                combined_results[doc_id]["semantic_score"] = normalized_score
        
        for result in combined_results.values():
            result['hybrid_score'] = hybrid_score(result["bm25_score"],result["semantic_score"], alpha)
            
        sorted_results = sorted(
            combined_results.values(),
            key=lambda result:result["hybrid_score"],
            reverse=True
        )
        
        return sorted_results
        

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)
        combined_results = {}
        
        for rank, (doc_id, score) in enumerate(bm25_results, start=1):
            
            combined_results[doc_id] = {
                "document": self.idx.docmap[doc_id],
                "bm25_rank": rank,
                "semantic_rank": None,
                "rrf_score": rrf_score(rank,k)
            }
        
        for rank, result in enumerate(semantic_results, start=1):
            
            doc_id = result["id"]
            
            if doc_id not in combined_results:
                combined_results[doc_id] = {
                    "document": self.documents[
                        next(i for i, doc in enumerate(self.documents)
                             if doc["id"] == doc_id)
                    ],
                    "bm25_rank": None,
                    "semantic_rank": rank,
                    "rrf_score": rrf_score(rank, k)
                }
            else:
                combined_results[doc_id]["semantic_rank"] = rank
                combined_results[doc_id]["rrf_score"] += rrf_score(rank,k)
        
        sorted_results = sorted(
            combined_results.values(),
            key=lambda result:result["rrf_score"],
            reverse=True
        )
        
        return sorted_results[:limit]
    
def normalize_command(scores: list[float]):
    if not scores:
        return

    max_score = max(scores)
    min_score = min(scores)
    denom = max_score - min_score

    if denom == 0:
        normalized = [1.0 for _ in scores]
    else:
        normalized = [
            (score - min_score) / denom
            for score in scores
        ]

    for score in normalized:
        print(f"* {score:.4f}")

def weighted_search_command(query, alpha, limit):
    documents = load_movies()

    engine = HybridSearch(documents)

    results = engine.weighted_search(
        query,
        alpha,
        limit
    )

    for i, result in enumerate(results[:limit], start=1):
        doc = result["document"]

        print(f"{i}. {doc['title']}")
        print(f"  Hybrid Score: {result['hybrid_score']:.3f}")
        print(
            f"  BM25: {result['bm25_score']:.3f}, "
            f"Semantic: {result['semantic_score']:.3f}"
        )
        print(f"  {doc['description'][:100]}...")

def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_search_command(
    query, k, limit, enhance=None, rerank_method=None
):
    search_query = query

    # Query enhancement
    if enhance == "spell":
        enhanced_query = correct_spelling(query)
    elif enhance == "rewrite":
        enhanced_query = rewrite(query)
    elif enhance == "expand":
        enhanced_query = expand(query)
    else:
        enhanced_query = query

    if enhanced_query != query:
        print(
            f"Enhanced query ({enhance}): "
            f"'{query}' -> '{enhanced_query}'\n"
        )

    search_query = enhanced_query

    # Load documents and initialize search engine
    documents = load_movies()
    engine = HybridSearch(documents)

    # Retrieve 5x candidates when individual reranking is enabled
    rerank_limit = limit * 5 if rerank_method else limit

    results = engine.rrf_search(search_query, k, rerank_limit)

    # Apply LLM reranking
    if rerank_method in ("individual", "batch", "cross_encoder"):
        print(
            f"Re-ranking top {len(results)} results "
            f"using {rerank_method} method..."
        )

        results = rerank(results, search_query, rerank_method)

    # Display only the requested number of results
    for i, result in enumerate(results[:limit], start=1):
        doc = result["document"]

        print(f"\n{i}. {doc['title']}")

        if rerank_method == "individual":
            print(f"  Re-rank Score: {result['rerank_score']:.3f}/10")

        if rerank_method == "cross_encoder":
            print(f"  Cross Encoder Score: {result['cross_encoder_score']:.3f}")

        print(f"  RRF Score: {result['rrf_score']:.3f}")
        print(
            f"  BM25 Rank: {result['bm25_rank']}, "
            f"Semantic Rank: {result['semantic_rank']}"
        )
        print(f"  {doc['description'][:100]}...")