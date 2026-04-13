# Task 1.2: Chunking Strategy Implementation — Design Decision

---

## Overview

After parsing all 75 documents into Markdown format (Task 1.1), the next step is to split the documents into smaller units ("chunks") suitable for embedding and retrieval. This writeup describes the two chunking strategies implemented, the preprocessing decisions made, and the rationale behind key configuration choices.

---

## Preprocessing Before Chunking

Before applying any chunking strategy, two special cases from the Marker/MinerU parsed output required handling:

### 1. Image Placeholders

Marker and MinerU insert figure references as Markdown image syntax:
```
![caption](images/_page_3_Figure_1.jpeg)
```

These were replaced with semantic text placeholders:
```
[FIGURE: caption]
```

**Rationale:** Raw image paths are meaningless to an embedding model and would introduce noise into chunk vectors. Replacing them with descriptive placeholders preserves the contextual signal (e.g., "this chunk discusses a diagram of the attention mechanism") without referencing a broken file path in the vector store.

### 2. LaTeX Formula Protection

Mathematical expressions appear in two forms in the parsed output:

**Block formulas:**
```
$$p_0^2 + p_1^2 + \dots + p_{s_i}^2 = C_i \tag{3}$$
```

**Inline formulas:**
```
$f^m(u) = (p_1 + \cdots p_{e_k})$
```

Both forms were temporarily marked as atomic units before chunking:
```
$$LATEX_BLOCK_START ... LATEX_BLOCK_END$$
[LATEX_INLINE: ...]
```

After chunking, all markers were restored to their original LaTeX format.

**Rationale:** A formula split across two chunk boundaries renders both chunks semantically meaningless. For example, splitting `$$p_0^2 + p_1^2` into one chunk and `= C_i$$` into another produces two incomplete, unembeddable expressions. Protecting formulas as atomic units ensures mathematical content is always retrievable as a coherent unit.

---

## Why Chunk Size and Overlap Matter

**Chunk size** controls the granularity of retrieval:
- Too large (e.g., 1024+ tokens): chunks contain multiple topics, producing unfocused embeddings that retrieve partially relevant content
- Too small (e.g., < 100 tokens): chunks lack surrounding context, making isolated sentences ambiguous or uninterpretable
- **512 tokens** was selected as the target size — well-suited for academic papers where a single argument or concept is typically developed over one to three paragraphs

**Overlap** prevents information loss at chunk boundaries:
- When a document is split at a hard boundary, a key sentence near that boundary may appear incomplete in both adjacent chunks
- An overlap of **50 tokens** ensures content near chunk boundaries appears in both the preceding and following chunks, preserving continuity without significantly increasing total chunk count

---

## Strategy 1 — Recursive Character Text Splitting

### Description

The recursive splitter attempts to split text by trying a hierarchy of separators in order:
```
\n\n → \n → . → (space) → character
```

It recurses through this hierarchy until all chunks fall within the target token limit. Token counting used the **cl100k_base** tokenizer (same as OpenAI embeddings) to ensure consistency with the downstream embedding model.

**Configuration:**
```
chunk_size    = 512 tokens
chunk_overlap = 50 tokens
```

### Results

| Metric | Value |
|---|---|
| Total chunks | 5,495 |
| Mean chunk length | 391.7 tokens |
| Median chunk length | 444.0 tokens |
| Std deviation | 130.1 tokens |
| Min chunk length | 20 tokens |
| Max chunk length | 513 tokens |
| Mean chunks per document | 74.6 |
| Median chunks per document | 57.0 |
| Min / Max chunks per doc | 13 / 696 |

### Strengths
- Fast and language-agnostic
- Produces consistent, predictable chunk sizes
- Strong baseline for general-purpose retrieval

### Limitations
- Structure-blind: does not know whether it is cutting at a section boundary, mid-argument, or mid-equation
- May split a paragraph mid-sentence when no natural separator falls within the token limit

---

## Strategy 2 — Section-Aware Chunking

### Description

Academic papers parsed by Marker and MinerU are output in Markdown format, where section headers are explicitly marked with `#`, `##`, and `###`. Section-aware chunking leverages this structure by using headers as natural chunk boundaries.

Each Markdown section is extracted as a candidate chunk. If a section fits within 512 tokens it is stored as a single chunk. If it exceeds the limit (e.g., a long Results or Appendix section), it is further split using the recursive splitter as a fallback.

**Configuration:**
```
max_section_tokens = 512
fallback           = RecursiveCharacterTextSplitter (same config as Strategy 1)
```

### Results

| Metric | Value |
|---|---|
| Total chunks | 6,118 |
| Mean chunk length | 339.7 tokens |
| Median chunk length | 401.0 tokens |
| Std deviation | 158.7 tokens |
| Min chunk length | 20 tokens |
| Max chunk length | 513 tokens |
| Mean chunks per document | 85.6 |
| Median chunks per document | 66.0 |
| Min / Max chunks per doc | 15 / 712 |

### Strengths
- Each chunk tends to correspond to one coherent topic (e.g., "Related Work", "Methodology", "Ablation Study")
- Preserves the logical structure of the paper within each chunk
- Better suited for queries targeting a specific section of a paper

### Limitations
- Higher variance in chunk sizes due to uneven section lengths (short "Abstract" vs. long "Experiments")
- Dependent on consistent header formatting in the parsed output — documents where Marker/MinerU failed to detect headers fall back entirely to recursive splitting

---

## Strategy Comparison

| Metric | Recursive | Section-Aware |
|---|---|---|
| Total chunks | 5,495 | 6,118 |
| Mean tokens | 391.7 | 339.7 |
| Median tokens | 444.0 | 401.0 |
| Std deviation | 130.1 | 158.7 |
| Mean chunks/doc | 74.6 | 85.6 |

Section-aware produces more chunks with higher variance, reflecting the uneven length of paper sections. Recursive produces fewer, more uniform chunks but without structural awareness.

**Expected retrieval behavior:**
- Section-aware is expected to perform better on queries targeting a specific section (e.g., "what evaluation metrics were used?" → maps to a Results/Evaluation section chunk)
- Recursive is expected to perform more consistently across general queries due to uniform chunk density

---

## Corpus Statistics (Post-Chunking)

| Metric | Recursive | Section-Aware |
|---|---|---|
| Raw chunks produced | 5,598 | 6,420 |
| Chunks after filtering (≥20 tokens) | 5,495 | 6,118 |
| Chunks removed | 103 | 302 |

The higher number of filtered chunks in section-aware reflects short sections such as "Acknowledgements", "Author Contributions", and appendix headers that contain minimal retrievable content.

---

## Output Files

| File | Description |
|---|---|
| `chunks_recursive_clean.json` | 5,495 cleaned chunks from recursive splitting |
| `chunks_section_clean.json` | 6,118 cleaned chunks from section-aware splitting |

Each chunk record contains:
```json
{
  "chunk_id"    : "2412.07587v6_rec_0",
  "arxiv_id"    : "2412.07587v6",
  "strategy"    : "recursive",
  "chunk_index" : 0,
  "text"        : "...",
  "token_count" : 387
}
```