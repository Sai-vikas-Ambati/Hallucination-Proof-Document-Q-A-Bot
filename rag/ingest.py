import os
import argparse
from pathlib import Path
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
import sys

# Ensure rag module can be imported if run directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rag.config import CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_PERSIST_DIR, EMBEDDING_MODEL_NAME

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    # Simple chunking by character, trying to split on boundaries
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        if end >= text_len:
            chunks.append(text[start:text_len])
            break
            
        boundary = text.rfind("\n\n", start, end)
        if boundary == -1 or boundary <= start + chunk_size // 2:
            boundary = text.rfind(". ", start, end)
        if boundary == -1 or boundary <= start + chunk_size // 2:
            boundary = text.rfind(" ", start, end)
            
        if boundary != -1 and boundary > start:
            chunks.append(text[start:boundary+1].strip())
            start = boundary + 1 - overlap
        else:
            chunks.append(text[start:end].strip())
            start = end - overlap
            
        if start < 0: start = 0
        if start == end: start = end + 1 # Prevent infinite loop
        
    return [c for c in chunks if c]

from chromadb.api.types import EmbeddingFunction

class EmbeddingFunctionWrapper(EmbeddingFunction):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(input, normalize_embeddings=True)
        return embeddings.tolist()
    def name(self) -> str:
        return "custom_bge_small"

def ingest_pdfs(pdf_dir: str):
    print(f"Initializing ChromaDB at {CHROMA_PERSIST_DIR}...")
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    
    emb_fn = EmbeddingFunctionWrapper(EMBEDDING_MODEL_NAME)
    collection = client.get_or_create_collection(
        name="financial_docs",
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Check already indexed docs
    existing_metadata = collection.get(include=["metadatas"])["metadatas"]
    indexed_docs = set()
    if existing_metadata:
        for m in existing_metadata:
            if m and "doc_name" in m:
                indexed_docs.add(m["doc_name"])
                
    pdf_paths = list(Path(pdf_dir).glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {pdf_dir}")
        return

    for pdf_path in pdf_paths:
        doc_name = pdf_path.stem
        if doc_name in indexed_docs:
            print(f"Skipping already indexed doc: {doc_name}")
            continue
            
        print(f"Processing: {doc_name}")
        doc = fitz.open(pdf_path)
        
        chunk_ids = []
        documents = []
        metadatas = []
        
        chunk_counter = 0
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if not text.strip():
                continue
                
            page_idx = page_num + 1 # 1-indexed
            
            # Chunk WITHIN each page
            chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            
            for chunk in chunks:
                prefix = f"[{doc_name} | page {page_idx}]"
                embedded_text = f"{prefix} {chunk}"
                
                chunk_id = f"{doc_name}_p{page_idx}_c{chunk_counter}"
                
                chunk_ids.append(chunk_id)
                documents.append(embedded_text)
                metadatas.append({
                    "doc_name": doc_name,
                    "page": page_idx,
                    "chunk_id": chunk_id,
                    "text_snippet": chunk
                })
                chunk_counter += 1
                
        if chunk_ids:
            collection.add(
                ids=chunk_ids,
                documents=documents,
                metadatas=metadatas
            )
            print(f"Added {len(chunk_ids)} chunks from {doc_name}")
            
    print("Ingestion complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB")
    parser.add_argument("pdf_dir", type=str, nargs='?', default=None, help="Directory containing PDFs")
    args = parser.parse_args()
    
    pdf_dir_to_use = args.pdf_dir if args.pdf_dir else os.environ.get("PDF_DIR", "../data/pdfs")
    
    # Correct path resolution if called without args
    if not args.pdf_dir:
        base_path = Path(__file__).parent.parent
        pdf_dir_to_use = str((base_path / "data" / "pdfs").resolve())
        
    ingest_pdfs(pdf_dir_to_use)
