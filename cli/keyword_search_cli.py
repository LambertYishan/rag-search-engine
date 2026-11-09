import argparse
import math
from lib.keyword_search import InvertedIndex, tokenize_text, tokenize_term, build_command, bm25_idf_command, bm25_tf_command, BM25_K1, BM25_B

    
def matches(query: str, title: str) -> bool:
    query_tokens = tokenize_text(query)
    title_tokens = tokenize_text(title)

    return any(
        query_token in title_token
        for query_token in query_tokens
        for title_token in title_tokens
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands"
    )

    search_parser = subparsers.add_parser(
        "search",
        help="Search movies using keywords"
    )
    
    build_parser = subparsers.add_parser(
        "build",
        help="Build and save the inverted index"
    )

    search_parser.add_argument(
        "query",
        type=str,
        help="Search query"
    )
    
    tf_parser = subparsers.add_parser(
        "tf",
        help="Get term frequency for a document"
    )
    
    tf_parser.add_argument(
        "doc_id",
        type=int,
        help="Document ID"
    )
    
    tf_parser.add_argument(
        "term",
        type=str,
        help="Search term"
    )
    
    idf_parser = subparsers.add_parser(
        "idf",
        help="Get the inverse document frequency of a word"
    )
    
    idf_parser.add_argument(
        "term",
        type=str,
        help="Search term"
    )
    
    tfidf_parser = subparsers.add_parser(
        "tfidf",
        help="Get the inverse document frequency of a word"
    )
    
    tfidf_parser.add_argument(
        "doc_id",
        type=int,
        help="Document ID"
    )
    
    tfidf_parser.add_argument(
            "term",
            type=str,
            help="Search term"
        )
    
    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")
    
    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )
    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of search results (default: 5)"
    )

    args = parser.parse_args()

    match args.command:
        case "search":
            index = InvertedIndex()
            
            try:
                index.load()
            except FileNotFoundError as err:
                print(err)
                return
            
            print(f"Searching for: {args.query}")
            
            query_tokens = tokenize_text(args.query)
            
            result_ids = set()
        
            for token in query_tokens:
                docs = index.get_document(token)
                
                for doc_id in docs:
                    result_ids.add(doc_id)
                    
                    if len(result_ids) == 5:
                        break
                if len(result_ids) == 5:
                    break
                
            for i, doc_id in enumerate(sorted(result_ids), start=1):
                movie = index.docmap[doc_id]
                print(f"{i}. {movie['title']}")
        
        case "build":
            build_command()
        
        case "tf":
            index = InvertedIndex()
            
            try:
                index.load()
            except FileNotFoundError as err:
                print(err)
                return
            
            term = tokenize_term(args.term)
            
            tf = index.get_tf(args.doc_id, term)
            
            print(tf)
        
        case "idf":
            index = InvertedIndex()
            
            try:
                index.load()
            except FileNotFoundError as err:
                print(err)
                return
            
            search_term = tokenize_term(args.term)
            
            N = len(index.docmap)
            DF = len(index.get_document(search_term))
            
            term_idf = math.log((N + 1) / (DF + 1))
            
            print(f"Inverse document frequency of '{args.term}': {term_idf:.2f}")

        case "tfidf":
            index = InvertedIndex()
            
            try:
                index.load()
            except FileNotFoundError as err:
                print(err)
                return
            
            search_term = tokenize_term(args.term)
            
            tf = index.get_tf(args.doc_id, search_term)
            
            N = len(index.docmap)
            DF = len(index.get_document(search_term))
            
            term_idf = math.log((N + 1) / (DF + 1))
            
            tf_idf = tf * term_idf
            
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        
        case "bm25idf":
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError as err:
                print(err)
                return
            
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
            
        case "bm25tf":
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError as err:
                print(err)
                return
            
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        
        case "bm25search":
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError as err:
                print(err)
                return
            
            results = index.bm25_search(args.query, args.limit)
            
            for i, (doc_id, score) in enumerate(results, start = 1):
                movie = index.docmap[doc_id]
                print(f"{i}. ({doc_id}) {movie['title']} - Score: {score:.2f}")
            
            
        case _:
            parser.print_help()
    
    

if __name__ == "__main__":
    main()
