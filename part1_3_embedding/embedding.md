# Task 1.3: Embedding & Vector Store — Design Decision

---

## Overview

After chunking all 75 documents into 5,495 (recursive) and 6,118 (section-aware) chunks, the next step is to convert each chunk into a dense vector embedding and store them in a vector database for similarity-based retrieval. This writeup describes the embedding model and vector database selected, the rationale behind each choice, and the verification results.

---

## Embedding Model: BAAI/bge-small-en-v1.5

### Description

BGE (BAAI General Embedding) is an open-source embedding model family developed by the Beijing Academy of Artificial Intelligence (BAAI), specifically fine-tuned for retrieval tasks using contrastive learning.

**Model specifications:**
```
Model    : BAAI/bge-small-en-v1.5
Dimensions: 384
Max tokens: 512
License  : MIT (free for commercial use)
Size     : ~90MB
```

### Why BGE-small

**Performance:** BGE models consistently rank highly on the MTEB (Massive Text Embedding Benchmark) retrieval leaderboard, outperforming general-purpose models such as `all-MiniLM-L6-v2` on passage retrieval tasks. Unlike sentence-transformers trained on general NLI data, BGE is specifically optimized for the retrieve-then-read pattern used in RAG pipelines.

**Dimensionality trade-off:** bge-small produces 384-dimensional vectors. Compared to larger alternatives:

| Model | Dimensions | Notes |
|---|---|---|
| BAAI/bge-small-en-v1.5 | 384 | Selected — fast, free, retrieval-optimized |
| BAAI/bge-large-en-v1.5 | 1024 | Higher accuracy, but 3x memory usage |
| text-embedding-3-small (OpenAI) | 1536 | Strong performance, but paid API |
| text-embedding-ada-002 (OpenAI) | 1536 | Widely used baseline, but paid API |
| all-MiniLM-L6-v2 | 384 | Same size, but not retrieval-optimized |

384 dimensions reduces storage requirements and speeds up nearest-neighbor search — important when indexing ~11,600 chunks across two collections — while maintaining strong retrieval performance for a 75-paper corpus.

**Cost:** The model runs fully locally with no API cost, making it suitable for Colab environments without GPU memory constraints.

**Trade-off acknowledged:** bge-small sacrifices some accuracy compared to bge-large or OpenAI embeddings. For a 75-paper academic corpus this trade-off is acceptable, but for production-scale retrieval a larger model would be preferred.

### Encoding Convention

BGE requires different prefixes for passages and queries:

```python
# Document (passage) encoding
"Represent this sentence: {chunk_text}"

# Query encoding  
"{query_text}"  # no prefix
```

All embeddings were L2-normalized (`normalize_embeddings=True`) to enable cosine similarity search.

---

## Vector Database: ChromaDB

### Description

ChromaDB is an open-source embedding database designed for AI applications. It stores vectors alongside document text and metadata, and supports similarity search using HNSW indexing.

### Why ChromaDB

**Zero configuration:** ChromaDB requires no external server, Docker setup, or cloud account. It runs fully in-process and persists data to a local directory, making it immediately usable in a Colab environment.

**Persistent storage:** By pointing ChromaDB to a Google Drive path, the index survives Colab session resets:
```python
chromadb.PersistentClient(
    path="/content/drive/MyDrive/[3Q]Capstone1/Assignment3/chromadb"
)
```

**Indexing method:** ChromaDB uses **HNSW (Hierarchical Navigable Small World graphs)** for approximate nearest-neighbor search. HNSW offers sub-linear query time O(log n) with high recall, making it efficient for collections of this size.

**Cosine similarity:** Collections were configured with `hnsw:space = cosine`, appropriate for normalized embedding vectors and standard for semantic retrieval tasks.

**Metadata filtering:** ChromaDB natively supports metadata storage alongside vectors. Each chunk was stored with:

```json
{
  "arxiv_id"    : "2412.07587v6",
  "strategy"    : "recursive",
  "chunk_index" : 42,
  "token_count" : 387
}
```

This enables source-level attribution and paper-level filtering in downstream retrieval.

**Trade-off acknowledged:** ChromaDB is not optimized for production scale (millions of vectors). For larger corpora, Qdrant or Weaviate would offer better performance, horizontal scaling, and more advanced filtering. For this 75-paper corpus, ChromaDB is the appropriate choice.

### Vector Database Comparison

| Database | Selected | Reason |
|---|---|---|
| ChromaDB | ✅ | Zero config, persistent, HNSW, metadata support |
| FAISS | No | In-memory only, no built-in persistence or metadata |
| Qdrant | No | Better for production scale, Docker required |
| Pinecone | No | Managed cloud service, free tier too limited |
| Weaviate | No | Overkill for this scale, complex setup |

---

## Index Statistics

| Collection | Chunks Stored | Embedding Dim | Similarity |
|---|---|---|---|
| recursive_chunks | 5,495 | 384 | Cosine |
| section_chunks | 6,118 | 384 | Cosine |

**Total vectors stored:** 11,613  
**Storage size:** 126.5 MB (chroma.sqlite3)  
**Storage location:** Google Drive → `[3Q]Capstone1/Assignment3/chromadb/`

---

## Index Verification — 3 Benchmark Queries

Three queries were sampled from the provided benchmark (`queries.json`) using their ground-truth document mappings from `qrels.json`. For each query, the top-5 most similar chunks were retrieved and checked for whether the ground-truth ArXiv paper appeared in the results (Hit@5).

Both collections successfully retrieved semantically relevant chunks, confirming that the embedding and indexing pipeline is functioning correctly.

### How to Access the Index

```python
import chromadb

client = chromadb.PersistentClient(
    path="/content/drive/MyDrive/[3Q]Capstone1/Assignment3/chromadb"
)

col_recursive = client.get_collection("recursive_chunks")
col_section   = client.get_collection("section_chunks")

print(col_recursive.count())  # 5,495
print(col_section.count())    # 6,118
```

---

## Output Files

| File | Description |
|---|---|
| `embedding_config.json` | Embedding model and vector DB configuration record |
| `chromadb/` (Google Drive) | Persistent ChromaDB index with all chunk vectors |