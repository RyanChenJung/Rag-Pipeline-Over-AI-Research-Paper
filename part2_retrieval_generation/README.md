# Part 2 — Retrieval & Generation: Technical Writeup

---

## Task 2.1 — Retrieval Pipeline

### Overview
The retrieval pipeline implements a `retrieve(query, k)` function dispatched across five strategies, all operating over two ChromaDB collections (`recursive_chunks` and `section_chunks`) backed by `BAAI/bge-small-en-v1.5` embeddings. All strategies return the top-k most relevant chunks, evaluated against ground-truth document mappings using Hit Rate@3 across all 496 benchmark queries.

---

### Strategy 1: Baseline Cosine Similarity
The baseline encodes each query using the same BGE embedding model used at index time and issues a nearest-neighbor lookup against ChromaDB. Results are converted from cosine distance to similarity scores via `score = 1 − distance`. This approach is computationally lightweight — a single embedding call followed by an approximate nearest-neighbor search — and serves as the natural lower-bound benchmark. Its primary limitation is that it encodes query and document independently, so queries using different vocabulary than the source passage may retrieve suboptimal results even when semantically aligned.

---

### Strategy 2: Re-Ranking with Cross-Encoder
Re-ranking addresses the independence limitation of bi-encoders by introducing a second-stage model that evaluates query-document pairs jointly. The implementation follows a two-stage pipeline: first, a larger candidate pool of `fetch_k = 20` chunks is retrieved via cosine similarity; each (query, chunk) pair is then scored by `cross-encoder/ms-marco-MiniLM-L-6-v2`, a MiniLM model fine-tuned on MS MARCO passage ranking. Candidates are re-sorted by cross-encoder score and the top k = 3 are returned.

The trade-off is latency: 20 forward passes through the cross-encoder per query versus 1 for baseline. However, the cross-encoder's token-level interaction modeling yields improved precision, particularly when query and passage phrasing diverges. Over section_chunks, re-ranking improved Hit Rate@3 from 0.9698 (baseline) to 0.9798 — a gain of 5 additional hits out of 496.

---

### Strategy 3: HyDE — Hypothetical Document Embedding 
HyDE inverts the retrieval framing: rather than embedding the raw question, it uses GPT-4o-mini to generate a hypothetical passage-style answer, then embeds that generated text for retrieval. The intuition is that a well-formed hypothetical answer written in academic register will occupy a closer position in the embedding space to actual paper passages than a short, interrogative query string.

A retry wrapper with 3-attempt exponential backoff handles transient API failures; if all attempts fail, the strategy falls back to embedding the original query. HyDE achieved Hit Rate@3 of 0.9536 on recursive_chunks and 0.9456 on section_chunks — notably underperforming the baseline on section_chunks (0.9698). This suggests that for this corpus, GPT-4o-mini's hypothetical answers occasionally diverge from the actual content structure of the ArXiv papers, steering retrieval toward plausible-sounding but mismatched regions of the embedding space. The added cost of one LLM call per query is therefore not justified in this setting.

---

### Strategy 4: Multi-Query Retrieval
Multi-query retrieval uses GPT-4o-mini to generate 3–5 query reformulations per input question. Each reformulation is embedded and issued independently against ChromaDB; the resulting candidate lists are merged and deduplicated by chunk ID, and the final top-k is selected from the merged pool. The motivation is that a single query is a point estimate of user intent — different phrasings can activate different regions of the embedding space, broadening recall.

In practice, 6 JSON parse failures occurred across 496 queries (approximately 1.2%), each falling back gracefully to the original query string. Multi-query achieved Hit Rate@3 of 0.9577 on recursive_chunks and 0.9778 on section_chunks. The section_chunks result represents a strong gain over baseline (+8 hits), confirming that query diversification is particularly effective when chunks carry meaningful section-level structure as retrieval anchors.

---

### Strategy 5: Hybrid BM25 + Dense Retrieval
The hybrid strategy combines sparse BM25 keyword retrieval (`rank_bm25`) with dense vector search using Reciprocal Rank Fusion (RRF). For each query, a BM25 ranked list and a dense cosine ranked list are produced independently over the same chunk corpus. RRF combines them by assigning each chunk a fused score of:

∑ 1 / (k + r_i), where r_i is the chunk's rank in list i and k = 60 is the standard smoothing constant.

The BM25 index is built once per collection and reused across all queries. The rationale is complementarity: BM25 excels at exact term matches (model names, ArXiv IDs, specific acronyms), while dense retrieval handles paraphrase and semantic similarity.

This combination yielded the best results of any strategy: 0.9758 on recursive_chunks and 0.9819 on section_chunks (487/496 hits), outperforming even re-ranking. Notably, hybrid retrieval is also the fastest advanced strategy — no LLM calls are required, and RRF is a lightweight rank fusion operation, making it highly efficient relative to its performance.

---

## Hit Rate@3 Results — All Configurations

| Strategy     | Collection        | Hit Rate@3 | Hits | Total |
|--------------|------------------|------------|------|-------|
| Baseline     | recursive_chunks  | 0.9496     | 471  | 496   |
| Baseline     | section_chunks    | 0.9698     | 481  | 496   |
| Re-ranked    | recursive_chunks  | 0.9698     | 481  | 496   |
| Re-ranked    | section_chunks    | 0.9798     | 486  | 496   |
| HyDE         | recursive_chunks  | 0.9536     | 473  | 496   |
| HyDE         | section_chunks    | 0.9456     | 469  | 496   |
| Multi-query  | recursive_chunks  | 0.9577     | 475  | 496   |
| Multi-query  | section_chunks    | 0.9778     | 485  | 496   |
| Hybrid       | recursive_chunks  | 0.9758     | 484  | 496   |
| Hybrid       | section_chunks    | 0.9819     | 487  | 496   |

---

## Key Observations
- Section-aware chunking consistently outperforms recursive chunking, indicating that structural document segmentation improves retrieval quality by preserving semantic boundaries.  
- Cross-encoder re-ranking improves performance over baseline dense retrieval, demonstrating the benefit of fine-grained relevance scoring after initial retrieval.  
- Hybrid retrieval achieves the best overall performance (0.9819 Hit Rate@3), confirming that lexical (BM25) and semantic signals are complementary.  
- HyDE does not consistently improve performance, likely due to noise introduced by LLM-generated hypothetical passages, which may drift from actual document vocabulary.  
- Multi-query retrieval improves recall, particularly for section-aware chunking, but provides diminishing returns compared to reranking and hybrid fusion.

---

## Conclusion
Overall, retrieval performance improves as we move from single-stage dense retrieval to multi-signal hybrid systems. The best-performing configuration combines section-aware chunking with hybrid BM25 + dense retrieval, suggesting that both document structure and retrieval diversity play critical roles in maximizing recall for scientific corpora.

Across all configurations, section_chunks consistently outperforms recursive_chunks, indicating that section-aware boundaries produce more coherent retrieval units than fixed-size recursive splits. The sole exception is HyDE, where section_chunks underperforms the baseline, a failure mode discussed above. Overall, these results reinforce the advantage of hybrid retrieval over section_chunks alone, and the best-performing hybrid configuration was therefore carried forward as the retrieval backend for Task 2.2.

---

# Task 2.2 — Generation with Grounded Prompting

## LLM Choice: GPT-4o-mini
GPT-4o-mini was selected as the generation model for three reasons. First, it reliably follows structured system prompts and citation formatting instructions — a key requirement given that every claim must be anchored to a [Source N] reference. Second, its cost-per-token is low enough to make full 496-query evaluation economically feasible. Third, at temperature = 0, GPT-4o-mini produces deterministic, factually conservative outputs that minimize unprompted generation beyond what the retrieved context supports. A larger model such as GPT-4o would offer higher reasoning quality but would not meaningfully improve grounded extractive answers where the constraint is context coverage rather than reasoning depth.

---

## System Prompt Design

You are a precise academic research assistant. Your job is to answer questions about AI/ML research papers.

STRICT RULES:
1. Answer ONLY using information from the provided context chunks below.
2. You MUST cite sources inline using [Source N] notation after every claim.
3. If the context does not contain enough information to answer, respond with exactly:
   "I don't have enough information in the provided context to answer this question."
4. Do NOT use any prior knowledge or make assumptions beyond what is stated in the context.
5. Keep answers concise and factual.
6. If multiple sources support a claim, cite all of them e.g. [Source 1][Source 2].

FORMAT:
- Write in clear prose
- Use [Source N] immediately after each claim
- End with a "Sources Used:" section listing which sources were cited

Each design element serves a specific purpose. Rule 1 is the primary anti-hallucination constraint. Without explicit restriction on prior knowledge, instruction-tuned LLMs frequently blend retrieved context with parametric memory — especially dangerous for a domain like ArXiv AI papers where the model has seen substantial training material. Rule 2 (mandatory inline citations) functions as both a grounding mechanism and an evaluable signal: being required to cite after every claim forces the model to locate a supporting passage before generating the statement. Rule 3 (exact fallback string) handles cases where retrieved context is genuinely insufficient, enabling programmatic detection of abstention during evaluation. The sanity check confirms this working correctly — queries like "How does incorporating demographic factors influence job transition predictions?" correctly return the fallback rather than fabricating an answer from the model's parametric knowledge about LLMs. The "Sources Used" section creates a structured audit trail for citation coverage analysis during Task 3.2 error analysis.

---

## Context Formatting
Each retrieved chunk is formatted as a labeled source block, placed before the question in the user message:

[Source 1: 2401.15478 — Introduction to XYZ]  
<chunk text, truncated to 1000 characters>

---

[Source 2: 2402.09721 — Related Work]  
<chunk text>

---

QUESTION: {query}

Remember: Answer only from the context above. Cite every claim with [Source N].

Positioning context before the question is consistent with the convention that background material should precede the task prompt, and keeps the retrieved passages adjacent to the beginning of the generation window. Chunk text is truncated to 1,000 characters to manage prompt length while preserving enough content for grounded answers. Three chunks at 1,000 characters each comfortably fit within GPT-4o-mini's context limit alongside the system prompt.

---

# Task 2.3 — Prompt Engineering Experiments

Four controlled experiments were run over 50 queries sampled from the benchmark, using section_chunks with re-ranked retrieval as the fixed configuration. Faithfulness and Answer Relevancy were scored using GPT-4o-mini as an LLM judge on a 0.0–1.0 scale, prompted to return a single decimal score. This approach was chosen over the full RAGAS framework to avoid dependency on spaCy and LangChain in the Colab environment while still producing quantitative, interpretable metrics across conditions. All results should be read as relative comparisons between conditions rather than as absolute scores, since the same model generating and scoring answers introduces a self-evaluation bias.

---

## Experiment 1: Minimal vs. Engineered Prompt

Hypothesis: An engineered prompt with explicit citation rules, a structured fallback, and format constraints produces more faithful and relevant answers than a bare minimum prompt.

Independent variable: System prompt complexity. The minimal prompt read: "You are a helpful assistant. Answer the question using the provided context." The engineered prompt was the full five-rule system prompt from Task 2.2. All other variables — queries, retrieved chunks, model, temperature — were held constant.

MINIMAL_SYSTEM = """You are a helpful assistant. Answer the question using the provided context."""


ENGINEERED_SYSTEM = """You are a precise academic research assistant answering questions about AI/ML research papers.


STRICT RULES:
1. Answer ONLY using information from the provided context chunks.
2. Cite sources inline using [Source N] notation after every claim.
3. If the context lacks sufficient information respond exactly: "I don't have enough information in the provided context to answer this question."
4. Do NOT use prior knowledge beyond what is in the context.
5. Keep answers concise and factual.

FORMAT:
- Write in clear prose
- Use [Source N] after each claim
- End with a "Sources Used:" section"""


| Condition  | Faithfulness | Answer Relevancy |
|------------|--------------|------------------|
| Minimal    | 0.9100       | 0.9360           |
| Engineered | 0.8460       | 0.8700           |

The results counter the hypothesis: the minimal prompt scored higher on both metrics. This is a meaningful and somewhat surprising finding. The most likely explanation is a self-evaluation artifact — the GPT-4o-mini judge applies similar grading logic to both conditions, and the minimal prompt produces more direct, flowing prose that the judge scores favorably. The engineered prompt, by constraining the model to add explicit [Source N] tags and a "Sources Used" section, may produce answers that appear more mechanical or incomplete to the judge even when they are better grounded. A secondary explanation is that for straightforward extractive questions, the minimal prompt is actually sufficient — the model's instruction-following defaults are strong enough that adding more rules produces marginal or negative returns on short-horizon, single-document queries. For longer, multi-hop queries the engineered prompt would likely be superior, but the 50-query sample may not have enough of these to show the effect.

---

## Experiment 2: Context Ordering (Lost in the Middle)

Hypothesis: Placing the most relevant chunk first improves faithfulness, consistent with the "lost in the middle" phenomenon where LLMs attend less reliably to content in the interior of long contexts.

Independent variable: Order of retrieved chunks in the context block — most-relevant-first (re-ranked order) versus most-relevant-last (reversed order).

| Condition | Faithfulness | Answer Relevancy |
|----------|--------------|------------------|
| Most relevant first | 0.8660 | 0.8500 |
| Most relevant last  | 0.5300 | 0.8260 |

This experiment produced the largest effect in the study. Reversing chunk order — so the highest-relevance chunk appears last — dropped faithfulness by 0.336 points (from 0.866 to 0.530), a drop of nearly 39%. Answer relevancy declined more modestly (0.024 points), suggesting that the model could still identify what to answer but lost its grounding in the source when the most relevant material was buried in the middle of the context block. This strongly confirms the "lost in the middle" hypothesis for this pipeline and empirically validates the design decision in Task 2.2 to serve chunks in re-ranked order. The implication is that retrieval order is not merely a cosmetic choice — it directly impacts generation faithfulness at a magnitude comparable to the choice of retrieval strategy itself.

---

## Experiment 3: Number of Retrieved Chunks (k = 3 vs. k = 5 vs. k = 10)

Hypothesis: Increasing k improves answer relevancy by providing more context coverage, but may degrade faithfulness by introducing noisy or off-topic chunks that distract the model.

Independent variable: Number of retrieved chunks passed to the LLM (k ∈ {3, 5, 10}).

| k  | Faithfulness | Answer Relevancy |
|----|--------------|------------------|
| 3  | 0.8960       | 0.8400           |
| 5  | 0.8900       | 0.8760           |
| 10 | 0.8960       | 0.8900           |

The results partially confirm the hypothesis but not in the expected direction. Faithfulness remained essentially flat across all k values (0.890–0.896), suggesting GPT-4o-mini is robust to context padding at the prompt lengths involved here. Answer relevancy, however, increased monotonically with k: 0.840 → 0.876 → 0.890. This indicates that additional retrieved chunks consistently add useful information rather than noise, at least up to k = 10. The likely explanation is that with only 3 chunks, some queries lack sufficient context coverage and the model must abstain or hedge; additional chunks fill those gaps. The faithfulness stability suggests the model remains well-grounded even at k = 10, meaning the instruction-following constraints from Task 2.2 are holding. Based on this, k = 5 offers a favorable trade-off between relevancy gains and prompt cost, though k = 10 is viable if token budget permits.

---

## Experiment 4: Chain-of-Thought vs. Direct Answering

Hypothesis: Chain-of-thought (CoT) prompting improves both faithfulness and answer relevancy by forcing the model to reason through retrieved evidence before producing a final answer, reducing the likelihood of unsupported claims.

Independent variable: Answering mode — direct (standard instruction prompt) versus chain-of-thought (prompt instructing the model to reason step-by-step before answering).

DIRECT_SYSTEM = """You are a precise academic research assistant answering questions about AI/ML research papers.
Answer ONLY from the provided context. Cite every claim with [Source N].
If context is insufficient, say: "I don't have enough information in the provided context to answer this question." """


COT_SYSTEM = """You are a precise academic research assistant answering questions about AI/ML research papers.


For EVERY question follow these steps explicitly:
STEP 1 — SCAN: Read all context chunks and identify which are relevant.
STEP 2 — EXTRACT: List the specific facts from relevant chunks that help answer the question.
STEP 3 — ANSWER: Write your final answer using ONLY extracted facts. Cite every claim with [Source N].
STEP 4 — VERIFY: Confirm every claim in your answer is directly supported by the context.


If at Step 2 you find no relevant facts respond: "I don't have enough information in the provided context to answer this question."
Do NOT use prior knowledge. Think step by step."""

| Condition | Faithfulness | Answer Relevancy |
|----------|--------------|------------------|
| Direct   | 0.8860       | 0.8500           |
| CoT      | 0.8960       | 0.8900           |

CoT outperformed direct answering on both metrics — faithfulness by 0.010 points and answer relevancy by 0.040 points. While the faithfulness gain is modest, the answer relevancy improvement is meaningful: CoT prompting appears to help the model identify more precisely what aspect of the question needs to be addressed before generating prose, producing more complete and targeted answers. This is consistent with the broader CoT literature showing that intermediate reasoning steps help models decompose complex questions. The improvement is not as large as the context ordering effect (Experiment 2), but it is consistently positive and comes at zero additional cost — only a prompt modification is needed. For production use, adding a CoT instruction to the system prompt from Task 2.2 would be a straightforward improvement.

---

## Summary Across All Experiments

| Experiment | Key Finding | Effect Size |
|------------|------------|-------------|
| Minimal vs. Engineered Prompt | Minimal prompt scored higher under judge evaluation | −0.066 faithfulness for engineered |
| Context Ordering | Most-relevant-last severely degrades faithfulness | −0.336 faithfulness |
| Chunk Count (k) | More chunks monotonically improve relevancy, no faithfulness cost | +0.050 relevancy from k=3 to k=10 |
| Chain-of-Thought | CoT consistently improves both metrics | +0.040 relevancy |

The dominant finding from Task 2.3 is that context ordering has a far larger impact on generation quality than prompt engineering choices. A poorly ordered context window (most relevant last) inflicts a faithfulness penalty larger than any gain achievable through prompt refinement. This reinforces the importance of the retrieval and re-ranking pipeline as a prerequisite for high-quality generation — the generation component is sensitive to what it receives, not just how it is instructed.
