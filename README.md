# RAG Pipeline over AI Research Papers

A retrieval-augmented generation (RAG) system for question answering over a corpus of 75 AI/ML research papers from ArXiv. 

**Team 1:** Ryan Chen, Sola Shin, Lawrence Lin, Juan de Haro

---

## Overview

This project implements an end-to-end RAG pipeline that ingests raw academic PDFs, converts them into structured, retrieval-ready chunks, and answers natural-language questions with inline citations back to source papers. The system was evaluated over a 496-query benchmark spanning text-only, text-image, text-table, and text-table-image question types, using the RAGAS framework to decompose performance into retrieval and generation quality.

**Best configuration:** Section-aware chunking + Hybrid BM25/Dense retrieval + Cross-encoder reranking, achieving:
- **Hit Rate@3:** 0.978
- **Faithfulness:** 0.792
- **Answer Correctness:** 0.483
- **Context Precision:** 0.920

## Architecture

```
PDF Corpus (75 papers)
        │
        ▼
┌───────────────────────┐
│  Parsing & Ingestion  │  Marker (primary) + MinerU (fallback)
│  → Markdown, LaTeX,   │  100% ingestion success rate
│    image placeholders │
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│  Preprocessing        │  Formula masking, image → semantic
│  & Sanitization       │  placeholder substitution, cl100k_base
│                       │  tokenization
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│  Chunking             │  Recursive (512 tok, 50 overlap) vs.
│                       │  Section-Aware (Markdown headers)
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│  Embedding            │  BAAI/bge-small-en-v1.5 (384-dim,
│                       │  L2-normalized)
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│  Vector Store         │  ChromaDB (persistent, HNSW,
│                       │  cosine distance)
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│  Retrieval            │  Hybrid BM25 + Dense (RRF, k=60)
│                       │  fetch_k=20 → top-k=3
│                       │  + optional cross-encoder reranking
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│  Generation           │  GPT-4o-mini (temp=0), context-only
│                       │  answering, mandatory [Source N]
│                       │  citations, fixed abstention string
└───────────────────────┘
```

## Key Components

| Component | Choice | Key Hyperparameters |
|---|---|---|
| PDF Parsing | Marker (primary) + MinerU (fallback) | — |
| Chunking | Recursive & Section-Aware (Markdown headers) | 512 tokens, 50-token overlap |
| Embedding model | BAAI/bge-small-en-v1.5 | 384 dims, L2-normalized |
| Vector database | ChromaDB (persistent) | cosine distance, HNSW |
| Retrieval strategy | Hybrid BM25 + Dense (RRF) | fetch_k=20, k=3, RRF k=60 |
| Re-ranking | ms-marco-MiniLM-L-6-v2 (cross-encoder) | fetch_k=20 → top-k=3 |
| Generation model | GPT-4o-mini | temperature=0, max_tokens=500 |
| Chunk truncation | 1,000 characters per chunk | — |

## Design Highlights

- **Hybrid parsing pipeline:** Marker handles the majority of double-column academic layouts via deep-learning layout detection; MinerU acts as a fallback for edge cases (e.g., recursive `TypeError`s), together achieving a 100% ingestion success rate across 75 papers.
- **Token-preserving sanitization:** LaTeX formulas are masked as atomic units before chunking to prevent equations from being split across chunk boundaries; image references are replaced with semantic placeholders (`[FIGURE: caption]`) to preserve contextual signal without polluting the embedding space.
- **Section-aware chunking:** Splits on Markdown header tags to align chunks with a paper's logical structure (e.g., isolating "Methodology" from "Related Work"), falling back to recursive splitting only when a section exceeds the token threshold. This consistently outperformed pure recursive chunking across all downstream metrics.
- **Hybrid retrieval:** Combines BM25 (exact term matching for model names, ArXiv IDs, acronyms) with dense cosine retrieval via Reciprocal Rank Fusion, selected as the production strategy for achieving the best Hit Rate@3 (0.982) with no added LLM calls.
- **Citation-grounded generation:** A five-rule system prompt enforces context-only answering, mandatory `[Source N]` inline citations, and an exact abstention string for programmatic detection of retrieval vs. generation failures during evaluation.

## Experimental Results

### Retrieval Strategy Comparison (Hit Rate@3, 496 queries)

| Strategy | Recursive Chunks | Section Chunks |
|---|---|---|
| Baseline (cosine) | 0.9496 | 0.9698 |
| Cross-encoder re-ranked | 0.9698 | 0.9798 |
| HyDE | 0.9536 | 0.9456 |
| Multi-query | 0.9577 | 0.9778 |
| **Hybrid (BM25 + Dense)** | 0.9758 | **0.9819** |

### RAGAS Aggregate Results

| Metric | Recursive + Baseline | Section + Baseline | Section + Hybrid | Section + Reranked |
|---|---|---|---|---|
| Faithfulness | 0.692 | 0.730 | 0.753 | **0.792** |
| Answer Relevancy | 0.623 | 0.688 | 0.715 | **0.748** |
| Context Precision | 0.850 | 0.860 | 0.895 | **0.920** |
| Context Recall | 0.742 | 0.737 | 0.754 | **0.782** |
| Answer Correctness | 0.387 | 0.433 | 0.448 | **0.483** |
| Hit Rate@3 | 0.950 | 0.968 | **0.982** | 0.978 |

### Prompt Engineering Findings

- **Minimal vs. engineered prompts:** The minimal prompt scored higher under an LLM judge, attributable to self-evaluation bias favoring citation-free prose over structured `[Source N]` outputs.
- **Context ordering:** Placing the most relevant chunk last (vs. first) dropped Faithfulness by 38.8% — the largest effect observed across all experiments — motivating a "most-relevant-first" context ordering in production.
- **Chunk count (k):** Faithfulness was stable across k=3/5/10; Answer Relevancy increased monotonically with more chunks, with no evidence of harmful noise at these lengths.
- **Chain-of-thought prompting:** Improved both Faithfulness and Answer Relevancy over direct prompting at zero additional cost, making it the recommended default.

## Known Limitations

- **Multimodal gap:** Answer Correctness drops from ~51% on text-only queries to ~35% on text-table-image queries. Figures and tables are currently represented as text placeholders rather than extracted content, affecting 167 of 496 benchmark queries (34%).
- **Document-level evaluation only:** Hit Rate@3 measures whether the correct *paper* was retrieved, not whether the specific answer-bearing *passage* was retrieved.
- **Judge bias:** Prompt engineering experiments were scored by a GPT-4o-mini judge, introducing self-evaluation bias; results should be treated as directionally indicative rather than absolute.
- **Fixed chunk truncation:** Chunks are truncated to 1,000 characters, which can discard relevant content in longer passages.

## Future Work

- Replace figure placeholders with rich captions generated by a vision-language model at ingestion time (Marker/MinerU already extract the underlying image files).
- Treat Markdown table blocks as atomic chunking units to prevent fragmentation of quantitative results tables.
- Add passage-level recall metrics alongside document-level Hit Rate@k.
- Replace LLM-judge scoring with the full reference-based RAGAS framework using a separate judge model.
- Explore dynamic, query-relevance-based chunk truncation in place of fixed character limits.

## Repository Structure

```
.
├── data_raw/                        # Raw source PDFs
├── parsed_output/                    # Markdown output from parsing stage
├── part1_1_parsing/
│   ├── parsing.md                     # Parsing write-up / notes
│   ├── parsing_marker_pymu.ipynb       # Marker (+ PyMuPDF) parsing pipeline
│   └── parsing_mineru.ipynb            # MinerU fallback parsing pipeline
├── part1_2_chunking/
│   ├── chunking.ipynb                   # Recursive & section-aware chunking
│   ├── chunking.md                       # Chunking write-up / notes
│   ├── chunks_recursive_clean.json        # Recursive chunk output
│   └── chunks_section_clean.json           # Section-aware chunk output
├── part1_3_embedding/
│   ├── embedding.ipynb                      # Embedding + ChromaDB indexing
│   ├── embedding.md                          # Embedding write-up / notes
│   └── embedding_config.json                  # Embedding model/index config
├── part2_retrieval_generation/
│   ├── Retrieval & Generation.md                # Retrieval & generation write-up
│   └── task2_retrieval_generation_rag.i...       # Retrieval/generation notebook
├── part3_2_error_analysis/                        # Error analysis
├── .gitignore
└── Retrieval & Generation.md
```

## Team

- Ryan Chen
- Sola Shin
- Lawrence Lin
- Juan de Haro
