import argparse
from lib.hybrid_search import(
    normalize_command,
    weighted_search_command,
    rrf_search_command
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    
    subparsers = parser.add_subparsers(
            dest="command",
            help="available commands"
        )
    
    normalize_parser = subparsers.add_parser(
        "normalize",
        help="normalize a list of numbers"
    )
    
    normalize_parser.add_argument(
        "numbers",
        nargs="*",
        type=float,
        help="a list of number to be normalized"
    )
    
    weighted_search_parser = subparsers.add_parser(
        "weighted-search",
        help="conduct a weighted search for a given prompt"
    )
    weighted_search_parser.add_argument(
        "query",
        type=str,
        help="a search prompt to be compared against the database"
    )
    weighted_search_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="the ratio beyond keyword and semantic search"
    )
    weighted_search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="top N of returned result default to 5"
    )
    rrf_search_parser = subparsers.add_parser(
        "rrf-search",
        help="conduct a reciprocal rank fusion search for a given prompt"
    )
    rrf_search_parser.add_argument(
        "query",
        type=str,
        help="a search prompt to be compared against the database"
    )
    rrf_search_parser.add_argument(
        "-k",
        type=int,
        default=60,
        help="the number that gives smoothing between top and bottom results"
    )
    rrf_search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="top N of returned result default to 5"
    )
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell","rewrite"],
        help="Query enhancement method",
    )
    
    
    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_command(args.numbers)
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case"rrf-search":
            rrf_search_command(args.query,args.k,args.limit, args.enhance)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()