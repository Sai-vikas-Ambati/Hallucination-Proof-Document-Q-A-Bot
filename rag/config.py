import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHROMA_PERSIST_DIR = "./chroma_db"
PDF_DIR = "./data/pdfs"

SIM_THRESHOLD_DEFAULT = 0.35
TOP_K_DEFAULT = 5
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Design rationale:
# - Chunk size/overlap: ~1000 characters with ~200 overlap allows capturing one or two full paragraphs of financial 
#   context without overflowing context windows or diluting retrieval precision.
# - Page-bounded chunks: Essential for strict citations. A chunk crossing pages makes it ambiguous where the data came from.
# - Embedding model: BAAI/bge-small-en-v1.5 is small enough to run locally without a GPU (important for offline hackathon constraints)
#   while providing state-of-the-art semantic search accuracy. Normalized embeddings with cosine similarity are used.
# - Three guardrail layers:
#   1. Retrieval gate: Pre-emptively rejects unanswerable questions if no relevant documents are found, saving LLM calls.
#   2. Prompt instructions: Strict system prompt enforces the model to output refusal if context is insufficient.
#   3. Post-check verification: Ensures the LLM actually cited the provided documents in its JSON output.
