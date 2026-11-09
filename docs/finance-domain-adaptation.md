# Finance Domain Adaptation: Credit Risk and Loan Underwriting

This note describes a possible next phase for the RAG search engine. It is a research and portfolio roadmap, not a description of functionality already present in the repository.

## Why keep this separate?

The current project is a retrieval-focused movie search prototype. A finance adaptation changes the domain corpus, document structure, evaluation criteria, and responsible-use requirements. Keeping this plan in a separate document preserves a clear README: demonstrated work stays distinct from proposed work.

A dedicated branch is useful once implementation begins, for example `feat/credit-risk-rag`. If the work grows into a substantially different product or modeling project, consider a separate repository and explain the relationship to this retrieval foundation.

## Define the task before choosing the model

“Credit risk analysis” can mean several different tasks:

1. **Evidence retrieval:** find relevant passages in policies, regulations, financial statements, or loan files.
2. **Evidence-grounded question answering:** summarize retrieved evidence with document and page citations.
3. **Risk signal extraction:** identify structured facts or risk indicators from source text.
4. **Default-risk prediction:** estimate an outcome from labeled historical examples.

RAG is a natural fit for the first two and can support the third. RAG alone does not establish predictive validity for default risk. If the objective is to predict repayment outcomes, treat that as a separate supervised-learning task with a properly defined target, temporal validation, and comparison to tabular baselines.

A compelling first increment would be a **credit-policy evidence assistant**: ask a question about an approved, public policy corpus; retrieve the relevant passages; return a concise answer with citations; and abstain when the corpus does not support an answer.

## Suggested implementation phases

### 1. Choose and document a narrow use case

Pick one user question type and one corpus, such as locating underwriting policy criteria or retrieving financial-statement definitions. State what the prototype can and cannot answer.

### 2. Build a traceable corpus

Start with public, permitted documents or clearly labeled synthetic examples. Preserve source title, publication date, page or section, document type, and URL or provenance. Avoid personal or confidential borrower data. Record document versions and data-processing steps.

### 3. Adapt ingestion and chunking

Financial documents contain tables, headings, footnotes, and references where layout carries meaning. Preserve section and page boundaries, represent tables deliberately, and attach metadata to every chunk. Compare chunk sizes and overlap on a fixed evaluation set instead of assuming one setting is best.

### 4. Compare retrieval methods

Use the current BM25, dense, and hybrid retrieval paths as baselines. Evaluate options such as metadata filters, query decomposition, and a domain-appropriate reranker only when the baseline evaluation shows a specific retrieval gap. Keep a simple, inspectable baseline for comparison.

### 5. Add grounded answer generation

Pass retrieved evidence to the generator with instructions to cite source identifiers and page or section locations. The answer should distinguish source facts from interpretation, surface conflicts between documents, and abstain when evidence is insufficient. Verify citations against retrieved passages.

### 6. Evaluate and analyze errors

Create a small, reviewed question set with expected source passages and answer criteria. Split development and final evaluation questions before tuning. Report retrieval and answer quality separately.

Suggested retrieval metrics:

- **Recall@k:** whether relevant evidence appears in the first k retrieved passages.
- **MRR:** how highly the first relevant passage ranks.
- **nDCG@k:** ranking quality when relevance has graded labels.

Suggested answer checks:

- Citation correctness and citation coverage.
- Faithfulness to retrieved evidence.
- Answer completeness and abstention behavior.
- Error categories, including stale policy, conflicting sources, table parsing, and unsupported claims.

Report the corpus composition, question-set construction, evaluator process, and limitations alongside metrics. Avoid presenting results from a tiny or synthetic set as evidence of real-world underwriting performance.

## If adding default-risk prediction

Keep prediction experiments distinct from the RAG answer-quality evaluation. Define the outcome and prediction horizon precisely, prevent label and time leakage, use temporal holdouts where appropriate, and compare against straightforward tabular baselines. Report calibration and threshold-dependent tradeoffs in addition to ranking metrics. Analyze performance across relevant groups only where data use is lawful, appropriate, and sufficiently supported. A language model’s explanation is not evidence that its prediction is calibrated or valid.

## Portfolio presentation

When this work exists, add a concise “Finance-domain experiment” section to the main README with a link to this document. Include only verified implementation details and measured results. The project story should make clear:

- Which components are implemented versus planned.
- What data was used and under what terms.
- Which retrieval and answer baselines were compared.
- How evaluation data was created and held out.
- What the system does when evidence is missing or contradictory.
- Which limitations prevent operational or underwriting use.

For a substantial implementation, work on a feature branch such as `feat/credit-risk-rag` and merge it only after the finance workflow and evaluation are independently reviewable. A separate repository is appropriate if it develops its own data pipeline, predictive modeling, or deployment lifecycle.

## Responsible-use boundary

This is an educational project plan. Real underwriting decisions involve legal, privacy, fairness, governance, and model-risk requirements that are outside the scope of a portfolio prototype. Do not use the system to make or recommend decisions about real applicants. Use appropriately licensed data, minimize sensitive information, and get qualified domain and compliance review before any consequential use.
