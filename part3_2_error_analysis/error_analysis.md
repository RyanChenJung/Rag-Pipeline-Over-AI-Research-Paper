# Task 3.2: Error Analysis — Design Decision

---

## Overview

This writeup analyzes the failure modes of our RAG pipeline across three configurations evaluated on the 496-query benchmark. The analysis identifies three worst-performing queries, one hallucination case, and proposes two concrete technical improvements grounded in the observed failures.

**Configurations analyzed:**

| Configuration | Chunking | Retrieval | Collection |
|---|---|---|---|
| baseline_recursive | Recursive (512 tokens) | Cosine similarity | recursive_chunks |
| baseline_section | Section-aware | Cosine similarity | section_chunks |
| reranked_section | Section-aware | Cosine + Re-ranking | section_chunks |

---

## Performance Overview

Answer-ground truth similarity scores by modality reveal a consistent pattern across all three configurations:

| Configuration | Overall | text | text-table | text-image | text-table-image |
|---|---|---|---|---|---|
| Baseline Recursive | 0.2059 | 0.2257 | 0.1976 | 0.1635 | 0.1470 |
| Baseline Section | 0.2125 | 0.2395 | 0.1615 | 0.1627 | 0.1410 |
| Reranked Section | 0.2179 | 0.2411 | 0.1935 | 0.1695 | 0.1608 |

**Key observation:** Text-only queries consistently outperform multimodal queries across all configurations. The 167 queries requiring visual or tabular content (33.7% of the benchmark) represent the primary performance gap in the pipeline.

---

## Failure Case 1 — Multimodal Query (Required)

**Query:** How do different biological pathway datasets compare in terms of distortion when using Euclidean versus mixed-curvature embeddings?

**Modality:** text-image

**System Answer (excerpt):**
> Different biological pathway datasets show varying levels of distortion when comparing Euclidean embeddings to mixed-curvature embeddings. The study indicates that mixed-curvature product embeddings generally yield lower distortion compared to Euclidean embeddings across all graphs in the five pathway datasets examined. Specifically, the results are illustrated in Figure 2(a), where points below t...

**Ground Truth:**
> Each dataset shows varying levels of distortion when comparing Euclidean and mixed-curvature embeddings. The differences highlight how each type of embedding handles specific characteristics inherent to each dataset's structure.

**Retrieved Documents:** `2401.15478v2` (all 3 chunks from same paper)

**Similarity Score:** 0.0073

### Failure Diagnosis: Retrieval Error — Multimodal Content Gap

The correct paper was successfully retrieved (Hit@3 = ✅). However, the answer requires reading **Figure 2(a)** — a bar chart comparing distortion scores across five biological pathway datasets — which is a visual artifact that the text-based index cannot access. During parsing, Marker extracted the figure as a JPEG file and inserted a `[FIGURE: ...]` placeholder in the Markdown. This placeholder was embedded as part of a text chunk, but carries no visual information.

The system attempted to answer from the figure caption text alone, producing a response that references "Figure 2(a)" and fabricates directional claims about the chart content. This is a **retrieval failure at the modality boundary**: the retriever found the right document but the index does not contain the visual information required to answer the query.

**Failure classification:** Retrieval error (modality gap)

---

## Failure Case 2 — Table-Dependent Query

**Query:** What are the key differences in performance between the Sylber models and the HuBERT base model across various speech processing tasks?

**Modality:** text-table

**System Answer:**
> I don't have enough information in the provided context to answer this question.
> Sources Used: None

**Ground Truth:**
> The Sylber models generally outperform the HuBERT base model in keyword spotting (KS) and intent classification (IC), but they show reduced effectiveness in speaker identification (SID), slot filling (SF), and automatic speech verification (ASV). This is attributed to challenges with handling lengthy inputs at regular sampling rates. In phoneme recognition, measured by PER, HuBERT performs better...

**Retrieved Documents:** `2410.07168v2` (all 3 chunks from same paper)

**Similarity Score:** 0.0225

### Failure Diagnosis: Retrieval Error — Table Content Fragmentation

The correct paper was retrieved (Hit@3 = ✅). The system correctly declined to hallucinate ("I don't have enough information"), which is appropriate behavior given the system prompt. However, the performance comparison data lives in a **results table** in the paper. During chunking, the Markdown table was likely split across chunk boundaries by the recursive splitter when the table exceeded the 512-token limit, or the table rows containing the specific task scores (KS, IC, SID, SF, ASV) were not captured in the retrieved chunks.

The retrieved chunks likely contained the paper's introduction and methodology sections rather than the results table, making it impossible for the LLM to produce the quantitative comparison required.

**Failure classification:** Retrieval error (table fragmentation during chunking)

---

## Failure Case 3 — Text-Only Simple Factual Query

**Query:** Does the foreground class distribution change between train and test sets?

**Modality:** text

**System Answer:**
> The context does not provide specific information about whether the foreground class distribution changes between the train and test sets. It mentions that the object distribution in the training set is similar to that in the test set for the LVIS v1 dataset, which implies some consistency, but it does not explicitly state the behavior of the foreground class distribution. Therefore, I don't have enough information to definitively answer this question.

**Ground Truth:** Yes.

**Retrieved Documents:** `2410.11774v2`, `2408.11208v3`, `2410.11774v2`

**Similarity Score:** 0.0122

### Failure Diagnosis: Generation Error — Over-Cautious Response to Simple Factual Query

This is a **generation error**, not a retrieval error. The relevant information was likely present in the retrieved chunks — the paper discusses LVIS v1 dataset properties where the foreground class distribution consistency between train and test sets is mentioned. However:

1. The system prompt instructs the model to say "I don't have enough information" when uncertain, which the model interpreted too broadly
2. The model over-hedged on a simple yes/no factual question, adding unnecessary qualifications ("implies some consistency", "does not explicitly state")
3. The ground truth is a direct "Yes" — a one-word answer that the model failed to produce despite having relevant context

This reflects a tension in the system prompt design: the "I don't have enough information" fallback, while useful for preventing hallucination, causes the model to under-answer clear factual questions that require only a direct extraction from context.

**Failure classification:** Generation error (over-cautious response)

---

## Hallucination Case

**Query:** Can media coverage contribute to hype in financial markets?

**Modality:** text

**Ground Truth:** Yes.

**System Answer (excerpt):**
> Yes, media coverage can contribute to hype in financial markets. A significant spike in the volume of news articles or social media mentions about a specific stock or sector, compared to its baseline, is identified as a key indicator of hype [Source 1]. This media coverage can lead to market overreactions, characterized by anomalies in price movement and volatility that deviate from historical patterns [Source 1]. Additionally, the imbalance in representation of news coverage relative to a stock...

### Hallucination Diagnosis

The ground truth is a simple "Yes." The system correctly answered "Yes" but then **over-generated** by adding several specific claims sourced from the retrieved paper (a study on hype-adjusted probability measures for NLP stock forecasting). While these claims are loosely grounded in the retrieved context, they represent an answer to a **different question** — specifically, "how does media coverage contribute to hype?" rather than "can it?"

The hallucination manifests as:
1. **Answer extension beyond the question scope**: the question asked for a binary yes/no; the system produced a multi-paragraph explanation
2. **Over-attribution**: specific claims about "significant spike in volume" and "market overreactions" are presented as direct answers rather than contextual detail from a specific paper
3. **Confidence inflation**: the response presents context-derived details as established facts without noting they are from one specific study

### Mitigation

Add explicit length and scope calibration to the system prompt:
```
If the question can be answered with a short factual response (yes/no, 
a number, a name, a date), provide that direct answer first. Only 
elaborate if the question explicitly asks for explanation or detail. 
Never extend your answer beyond what is directly required by the question.
```

---

## Proposed Improvements

### Improvement 1 — Multimodal Figure Captioning via Vision LLM

**Root cause addressed:** Failure Cases 1 and the broader text-image performance gap (115 queries, 23.2% of benchmark)

**Implementation:**

During the ingestion pipeline (extension of Task 1.1), pass each extracted figure image through a multimodal LLM (e.g., GPT-4o Vision) to generate a rich descriptive caption:

```python
from openai import OpenAI
import base64

def caption_figure(image_path: str, surrounding_text: str) -> str:
    """Generate detailed caption for a figure using GPT-4o Vision."""
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", 
                 "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                {"type": "text", 
                 "text": f"Describe this figure in detail for a technical RAG system. "
                         f"Context from paper: {surrounding_text[:500]}. "
                         f"Include all visible data, labels, axes, trends, and key values."}
            ]
        }]
    )
    return response.choices[0].message.content
```

Replace `[FIGURE: caption]` placeholders with the generated rich captions before embedding. This directly provides the visual information the retriever currently cannot access.

**Expected impact:** Significant improvement on text-image and text-table-image queries (167/496 = 33.7% of benchmark). The pipeline already extracts figure files (Marker and MinerU both output `images/` subfolders) so no changes to the parsing step are required.

---

### Improvement 2 — Atomic Table Chunking with Table-Type Metadata

**Root cause addressed:** Failure Case 2 and text-table query failures where results tables are fragmented during chunking

**Implementation:**

Extend the preprocessing step to detect and protect Markdown table blocks before chunking, similar to the LaTeX formula protection already in place:

```python
def protect_tables(text: str) -> str:
    """Mark Markdown tables as atomic units before chunking."""
    table_pattern = re.compile(
        r'(\|.+\|\n\|[-:| ]+\|\n(?:\|.+\|\n)+)',
        re.MULTILINE
    )
    return table_pattern.sub(
        lambda m: f'TABLE_BLOCK_START\n{m.group()}TABLE_BLOCK_END\n',
        text
    )

def restore_tables(text: str) -> str:
    text = text.replace('TABLE_BLOCK_START\n', '')
    text = text.replace('TABLE_BLOCK_END\n', '')
    return text
```

Additionally, store each table as a **dedicated chunk** with a `content_type: table` metadata tag:

```python
# In chunking pipeline: detect and index tables separately
table_chunks = extract_table_chunks(doc["text"])
for table_chunk in table_chunks:
    all_chunks.append({
        "chunk_id"     : f"{arxiv_id}_table_{idx}",
        "arxiv_id"     : arxiv_id,
        "content_type" : "table",   # enables table-targeted retrieval
        "text"         : table_chunk,
        "token_count"  : count_tokens(table_chunk)
    })
```

This ensures: (1) tables are never split mid-row, preserving the relational structure of tabular data; (2) table chunks can be targeted specifically when a query is detected as table-dependent (e.g., contains "compare", "performance", "results table").

**Expected impact:** Direct improvement on the 28 text-table queries in the benchmark. More broadly benefits any query requiring quantitative comparisons from experimental result tables, which are common in AI/ML papers.

---

## Summary

| Issue | Case | Failure Type | Proposed Fix |
|---|---|---|---|
| Figure content inaccessible | Case 1, most text-image | Retrieval (modality gap) | Vision LLM captioning (Improvement 1) |
| Table content fragmented | Case 2, text-table queries | Retrieval (chunking) | Atomic table protection (Improvement 2) |
| Over-cautious generation | Case 3 | Generation error | System prompt calibration |
| Over-generation hallucination | Hallucination case | Generation error | Length/scope prompt instruction |
