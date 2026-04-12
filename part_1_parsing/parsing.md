# Task 1.1: PDF Parsing & Preprocessing — Design Decision

## Overview

To parse the corpus of 75 AI/ML research papers, we evaluated three PDF parsing tools through direct empirical testing on the actual corpus: **Marker** (v1.10.2), **MinerU** (v2.x), and **PyMuPDF** (fitz). The final pipeline uses a **hybrid approach**: Marker as the primary parser for 57 documents, and MinerU as a fallback parser for the remaining 18 documents that Marker could not process.

---

## Tools Evaluated

### 1. Marker (Primary Parser — 57/75 documents)

Marker is a deep-learning-based PDF parser built on the Surya OCR stack. It performs layout detection, text recognition, table parsing, and formula conversion in a single pipeline.

**Strengths observed:**
- Correctly reconstructed multi-column academic layouts, preserving reading order across columns
- Converted mathematical formulas into LaTeX-compatible notation (e.g., `\mathcal{L}`, `\nabla`), enabling the LLM to process mathematical logic as structured text rather than corrupted Unicode
- Extracted tables into Markdown format (`|---|`), preserving row/column relationships for downstream chunking
- Automatically removed non-content elements such as headers, footers, and page numbers, reducing noise in the vector embeddings
- Extracted figures and generated image placeholders (`![](images/xxx.jpeg)`) directly in the Markdown output, providing a foundation for the multimodal bonus extension

**Limitations encountered:**
- `TypeError: function takes at most 16 arguments (17 given)`: 18 out of 75 PDFs failed with this error, caused by an incompatibility between Marker's internal Surya OCR function and certain complex PDF structures (dense mathematical proofs, nested theorem environments)
- Required careful dependency management: `torch==2.4.1+cu124`, `transformers==4.57.6`, `surya-ocr==0.17.1`, and `protobuf==3.20.3` had to be precisely pinned to avoid `ImportError: cannot import name 'PreTrainedModel'` and `ncclCommWindowDeregister` symbol errors on Colab's CUDA 12.4 environment

**Sample output quality (2406.15888v2.pdf):**
```
# **Real-time Speech Summarization for Medical Conversations**
## Abstract
## 1. Introduction
## 2. Real-time Speech Summarization System
```
- 48,128 characters extracted, 1 image placeholder, full table structure preserved

---

### 2. MinerU (Fallback Parser — 18/75 documents)

MinerU is an academic-focused document parser developed by the OpenDataLab team, designed specifically for scientific literature with dense mathematical content.

**Why it was chosen as fallback:**
- The 18 documents that failed Marker were mathematically intensive (e.g., theorem proofs with multi-line equation arrays, molecular generation models), precisely the use case MinerU is optimized for
- MinerU rendered LaTeX equations in `$$...$$` block format (e.g., `$$x_l = f(a_l)$$`), which is cleaner and more LLM-readable than Marker's inline notation for complex derivations
- Section headers were correctly identified and formatted as `##` Markdown headings

**Sample output quality (2412.12783v2.pdf):**
```
# Noise-based Local Learning using Stochastic Magnetic Tunnel Junctions
## Abstract
## I. INTRODUCTION
## II. METHODS
$$x_l = f(a_l)$$
$$\mathcal{L} = ||y^* - y||^2$$
```
- Clean LaTeX math blocks, correct heading hierarchy, 100% success rate on all 18 previously failed documents

**Limitations:**
- Heavier installation footprint than Marker; requires model download (~5–10 minutes on first run)
- Slower per-document processing time compared to Marker on standard academic papers

---

### 3. PyMuPDF / fitz (Evaluated, Not Used in Final Pipeline)

PyMuPDF is a coordinate-based text extraction library that does not use OCR or layout models.

**Why it was rejected:**
- No semantic structure recovery: section headers appeared as plain text with no Markdown formatting, making section-aware chunking impossible without additional heuristics
- Mathematical formulas were extracted as corrupted Unicode or whitespace (e.g., `PCCi = Cov(epi, ˆepi) / Var(epi)` instead of proper notation), rendering them useless for technical retrieval
- Table structure was not preserved; table content was linearized into plain text rows
- Column reading order was unreliable on two-column academic layouts, producing interleaved text from left and right columns

PyMuPDF was tested on the same 18 failed documents and produced syntactically valid output, but the semantic quality was insufficient for a technical RAG system where mathematical and structural accuracy directly impacts retrieval and generation quality.

---

## Final Parsing Architecture

| Documents | Parser | Reason |
|---|---|---|
| 57/75 | Marker v1.10.2 | Standard academic papers; full layout, table, formula, and image extraction |
| 18/75 | MinerU v2.x | Mathematically dense papers; Marker internal error on complex PDF structures |
| 0/75 | PyMuPDF | Evaluated but rejected due to insufficient structural and formula quality |

All 75 documents were successfully parsed. Output for each document is stored as:
```
parsed_output/
└── {arxiv_id}/
    ├── {arxiv_id}.md      # Full document in Markdown
    └── images/
        └── _page_N_Figure_M.jpeg  # Extracted figures
```

---

## Handling Tables

Both Marker and MinerU extract tables into Markdown table syntax, preserving row and column relationships. This was preferred over CSV extraction because:
- Markdown tables remain embedded in the document flow, preserving surrounding context (e.g., the paragraph referencing "Table 3" stays adjacent to the table data)
- Downstream chunking strategies can treat tables as coherent blocks rather than separate files requiring re-linking

---

## Handling Figures/Images

Marker extracts embedded figures as JPEG files into an `images/` subfolder and inserts Markdown placeholders (`![caption](images/file.jpeg)`) at the correct position in the text flow. MinerU follows the same convention. This provides the structural foundation for the **Multimodal Handling bonus extension**: figure placeholders in the Markdown can be replaced with LLM-generated captions by passing the image files through a multimodal model (e.g., GPT-4o Vision), after which the captions are embedded alongside the surrounding text chunks.

PyMuPDF can extract raw image bytes but does not insert placeholders into the text, making multimodal integration significantly more complex.

---

## Corpus Statistics (Post-Parsing)

| Metric | Value |
|---|---|
| Total documents parsed | 75 |
| Documents parsed by Marker | 57 |
| Documents parsed by MinerU | 18 |
| Total extracted characters | ~3.2M (est.) |
| Mean document length | ~42,000 characters |
| Total extracted images | ~320 figures |

---

## Alternatives Considered

| Tool | Rejected Reason |
|---|---|
| PyMuPDF (fitz) | No formula/table/structure recovery |
| pdfplumber | Better table extraction than PyMuPDF but still no formula support; no image extraction |
| Nougat (Meta) | Slower inference, higher hallucination rate on out-of-domain documents; no longer actively maintained |
| Adobe PDF Extract API | Commercial, cost-prohibitive at 75-document scale |
| Unstructured.io | Strong general-purpose parser but weaker LaTeX formula fidelity compared to Marker/MinerU |