# RAG Search Engine

A from-scratch search and retrieval project that explores how lexical and semantic methods can work together. The current implementation uses a movie-description corpus to compare keyword search, dense vector search, chunk-level semantic retrieval, and hybrid ranking from a command-line interface.

The project is designed to make core information-retrieval ideas inspectable: build an inverted index, calculate BM25 scores, encode text with a sentence-transformer, and combine retrieval signals. It is an applied portfolio project in information retrieval, NLP, and practical Python engineering.

> **Scope:** This repository currently demonstrates retrieval over movie data. It is not a credit-risk model, loan-underwriting system, or validated financial decision tool. Finance-domain adaptation is outlined in [docs/finance-domain-adaptation.md](docs/finance-domain-adaptation.md) as a proposed next step.

## What it does

- **Lexical retrieval:** builds an inverted index over movie titles and descriptions; tokenization includes punctuation cleanup, stopword removal, and Porter stemming.
- **BM25 ranking:** scores candidates using term frequency, inverse document frequency, and document-length normalization.
- **Dense semantic retrieval:** embeds movie descriptions and queries with the all-MiniLM-L6-v2 Sentence Transformers model and ranks by cosine similarity.
- **Chunk-level retrieval:** splits descriptions into overlapping semantic chunks, embeds passages, and retrieves at chunk granularity.
- **Hybrid retrieval:** combines lexical and semantic results with weighted score fusion or Reciprocal Rank Fusion (RRF).
- **Query enhancement:** optionally applies spelling correction or concise query rewriting through an OpenAI-compatible API endpoint configured for OpenRouter.
- **Command-line workflows:** exposes commands for building/searching indexes, inspecting retrieval components, and running semantic or hybrid searches.

## Retrieval flow

~~~text
Movie corpus (data/movies.json)
        │
        ├── Tokenize → inverted index → BM25 ─────────┐
        │                                               ├── weighted fusion / RRF → ranked results
        └── SentenceTransformer → embeddings → cosine ┘
                                    │
                                    └── optional chunk-level semantic retrieval

Optional query enhancement: spelling correction or rewrite via OpenRouter
~~~

## Repository layout

~~~text
.
├── data/
│   ├── movies.json                 # Movie records used by the search corpus
│   └── stopwords.txt               # Stopwords for lexical indexing
├── cli/
│   ├── keyword_search_cli.py       # Inverted index and BM25 commands
│   ├── semantic_search_cli.py      # Embedding, chunking, and semantic search
│   ├── hybrid_search_cli.py        # Weighted and RRF hybrid search
│   └── lib/                        # Retrieval implementations
├── pyproject.toml                  # Python project dependencies
└── uv.lock                         # Locked dependency resolution
~~~

Generated indexes and embeddings are cached under cache/ when the corresponding build/search paths run. They are local artifacts and can be regenerated from the corpus.

## Skills and concepts demonstrated

- **Information retrieval:** inverted indexes, term frequency, IDF, BM25, cosine similarity, and rank fusion.
- **NLP preprocessing:** tokenization, stopword filtering, stemming, text chunking, and sentence-transformer embeddings.
- **Search system design:** sparse/dense retrieval tradeoffs, document and chunk retrieval, tunable weighted fusion, and RRF.
- **Applied machine learning:** using a pretrained embedding model for semantic representation and retrieval.
- **Python engineering:** modular retrieval components, command-line tools, local caching, and environment-based API configuration.
- **LLM integration:** optional query correction and rewriting through an OpenAI-compatible client.

## Requirements

- Python **3.14 or newer** (as declared in pyproject.toml)
- [uv](https://docs.astral.sh/uv/) for dependency and environment management
- Internet access on first use to download the sentence-transformer model
- An OpenRouter API key only for optional query-enhancement commands

## Setup

~~~bash
git clone https://github.com/LambertYishan/rag-search-engine.git
cd rag-search-engine
uv sync
~~~

For query enhancement, create a local .env file:

~~~dotenv
OPENROUTER_API_KEY=your_key_here
~~~

Keep secrets out of source control. Search and embedding workflows that do not request query enhancement do not need this key.

## Try it

Run commands from the repository root:

~~~bash
# Build the lexical index
uv run python cli/keyword_search_cli.py build

# Search by words and phrases
uv run python cli/keyword_search_cli.py search "space adventure"

# Search by semantic similarity
uv run python cli/semantic_search_cli.py search "a family discovers a magical world" --limit 5

# Combine lexical and semantic scores
uv run python cli/hybrid_search_cli.py weighted-search "space adventure" --alpha 0.5 --limit 5

# Combine ranked lists with Reciprocal Rank Fusion
uv run python cli/hybrid_search_cli.py rrf-search "space adventure" --limit 5

# Rewrite a query before hybrid retrieval (requires OPENROUTER_API_KEY)
uv run python cli/hybrid_search_cli.py rrf-search "movie about a bear in london" --enhance rewrite --limit 5
~~~

Use --help on each CLI to see available subcommands and options. The first semantic search may take longer while the embedding model is downloaded and cached.

## Design notes

### Lexical and semantic retrieval

Lexical search is effective when queries share terms with the documents, especially for exact names and rare keywords. Semantic search can match related meaning and paraphrases, but may be less reliable on exact identifiers. Hybrid retrieval aims to benefit from both.

### Weighted fusion and RRF

Weighted fusion normalizes the retrieval scores and combines them using alpha:

~~~text
hybrid_score = alpha × lexical_score + (1 - alpha) × semantic_score
~~~

RRF combines ranked lists rather than requiring scores from different retrievers to share a scale. Its smoothing constant k controls how much rank differences affect the fused score. These options make retrieval behavior tunable and give the project a useful foundation for controlled comparison.

### Current evaluation limits

The repository includes retrieval components and command-line workflows, but this README does not claim benchmark performance. For a stronger ML/DS portfolio presentation, add a labeled query set and report metrics such as Recall@k, MRR, and nDCG@k; compare BM25, dense retrieval, weighted fusion, and RRF on the same fixed split. Record corpus size, query construction, parameter choices, and failure cases so results are reproducible.

## Roadmap

Potential next steps include a reproducible retrieval-evaluation set, experiment tracking for chunking and fusion parameters, richer metadata filters, and a finance-specific corpus and evaluation protocol. Finance-domain work should be treated as a separate experimental phase because it changes the corpus, risk taxonomy, labels, and validation requirements. See [the finance adaptation proposal](docs/finance-domain-adaptation.md).

## Responsible use

This project is an educational retrieval/search prototype. Its current movie corpus and retrieval outputs do not support financial, legal, or other consequential decisions. A future finance-oriented experiment would need appropriate data rights, clear label definitions, leakage controls, subgroup and calibration analysis, human review, and domain-specific validation before any decision-support claims.

## License

No license has been specified in this repository. Until one is added, reuse and redistribution permissions are not granted by default.
