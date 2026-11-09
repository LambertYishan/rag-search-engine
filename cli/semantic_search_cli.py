import argparse
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    semantic_search,
    chunk_text,
    semantic_chunk_text
)
from lib.chunked_semantic_search import (
    ChunkedSemanticSearch
)
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(
        dest="command",
        help="available commands"
    )

    subparsers.add_parser(
        "verify",
        help="print model specification"
    )

    embed_text_parser = subparsers.add_parser(
        "embed_text",
        help="embed a text input"
    )

    embed_text_parser.add_argument(
        "text",
        type=str,
        help="Text needed to be embedded"
    )
    
    verify_embeddings_parser = subparsers.add_parser(
        "verify_embeddings",
        help="verify embedding of documents"
    )
    
    embed_query_text_parser = subparsers.add_parser(
        "embed_query",
        help="embed a text input"
    )

    embed_query_text_parser.add_argument(
        "query_text",
        type=str,
        help="Query needed to be embedded"
    )
    
    semantic_search_parser = subparsers.add_parser(
        "search",
        help="embed a text input"
    )

    semantic_search_parser.add_argument(
        "search_term",
        type=str,
        help="term for semantic search"
    )
    semantic_search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="limit for top N results"
    )
    
    chunk_parser = subparsers.add_parser(
        "chunk",
        help="chunking the text"
    )

    chunk_parser.add_argument(
        "text",
        type=str,
        help="text to chunk"
    )
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="num of words to chunk, chunk-size defaulted to 200"
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=1,
        help="number of words overlapped in a chunk, defaulted to 1 and entered as 1"
    )
    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk",
        help="semantic chunking the text"
    )

    semantic_chunk_parser.add_argument(
        "text",
        type=str,
        help=" text to chunk"
    )
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=4,
        help="num of words to chunk, chunk-size defaulted to 4"
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="number of words overlapped in a chunk, defaulted to 0"
    )
    embed_chunks_parser = subparsers.add_parser(
        "embed_chunks",
        help="embed chunks"
    )
    search_chunked_parser = subparsers.add_parser(
        "search_chunked",
        help="search chunked from database"
    )
    search_chunked_parser.add_argument(
        "query",
        type=str,
        help=" text to search"
    )
    search_chunked_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="top result to show"
    )
    
    

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)
            
        case "verify_embeddings":
            verify_embeddings()
        
        case "embed_query":
            embed_query_text(args.query_text)
        
        case "search":
            semantic_search(args.search_term, args.limit)
        
        case "chunk":
            chunk_text(args.text, args.chunk_size,args.overlap)
        
        case "semantic_chunk":
            semantic_chunk_command(args.text, args.max_chunk_size, args.overlap)
                
        case "embed_chunks":
            embed_chunk_command()
            
        case "search_chunked":
            search_chunked_command(args.query, args.limit)

        case _:
            parser.print_help()

def load_movies():
    with open("data/movies.json", "r", encoding="utf-8") as f:
        movie_data = json.load(f)

    return movie_data["movies"]

def semantic_chunk_command(text, max_chunk_size, overlap):
    chunks = semantic_chunk_text(
        text,
        max_chunk_size,
        overlap
    )
    print(f"Semantically chunking {len(text)} characters")

    for i, chunk in enumerate(chunks, start=1):
        print(f"{i}. {chunk}")


def embed_chunk_command():

    engine = ChunkedSemanticSearch()

    with open("data/movies.json", "r", encoding="utf-8") as f:
        movie_data = json.load(f)

    documents = movie_data["movies"]

    embeddings = engine.load_or_create_chunk_embeddings(documents)

    print(f"Generated {len(embeddings)} chunked embeddings")

def search_chunked_command(query, limit):

    documents = load_movies()

    engine = ChunkedSemanticSearch()

    engine.load_or_create_chunk_embeddings(documents)

    results = engine.search_chunks(query, limit)

    # 5. Print results
    for i, result in enumerate(results, start=1):
        print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['document']}...")

if __name__ == "__main__":
    main()
