# Hallucination-Proof Document Q&A Bot

![App Screenshot](assets/screenshot.png)

A highly reliable RAG pipeline for financial PDFs built for hackathons. Prioritizes strict citations and robust guardrails to prevent hallucinations.

## Features
- **Strict Chunking**: Chunks never cross page boundaries to ensure exact citations.
- **Three-Layer Guardrail System**: 
  1. Retrieval gate (similarity threshold)
  2. Prompt-level strictness
  3. Post-generation JSON parsing & validation
- **Refusal Mechanism**: Unanswerable queries reliably return "I don't know based on the provided documents".
- **Local offline processing**: Everything except the Groq LLM API call runs locally (embeddings, vector store, UI).

## Tech Stack
- Python 3.11
- Streamlit (UI)
- ChromaDB (Local Vector Store)
- Sentence-Transformers (Embeddings: BAAI/bge-small-en-v1.5)
- PyMuPDF (PDF Parsing)
- Groq SDK (LLM Generation: llama-3.3-70b-versatile)

## Setup Instructions

1. **Install Requirements**
```bash
pip install -r requirements.txt
```

2. **Environment Setup**
Copy the sample env file and add your Groq API key:
```bash
cp .env.example .env
# Edit .env and insert your GROQ_API_KEY
```

3. **Add Data & Ingest**
Place your financial PDFs into the `data/pdfs/` directory (or upload them via the UI later).
Run the ingestion script:
```bash
python -m rag.ingest
```

4. **Run the App**
```bash
streamlit run app.py
```

## Architecture & Design Rationale
- **Chunking**: Chunks are limited to ~1000 characters with ~200 overlap to capture context without losing granularity. Bounding them strictly to single pages avoids citation ambiguity.
- **Embeddings**: `BAAI/bge-small-en-v1.5` is lightweight, doesn't require a GPU, and works fully offline.
- **Guardrails**: To ensure a hallucination-proof system, we reject out-of-context queries before they even hit the LLM (via ChromaDB distance), strictly prompt the LLM to output valid JSON with exact source IDs, and then perform a post-check to catch fake citations.

## Evaluation
Run the evaluation script to test refusal rates and citation hit rates:
```bash
python -m eval.run_eval
```
