import chromadb
import re
from pathlib import Path
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rag.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL_NAME, TOP_K_DEFAULT
from rag.ingest import EmbeddingFunctionWrapper

def get_db_collection():
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    emb_fn = EmbeddingFunctionWrapper(EMBEDDING_MODEL_NAME)
    return client.get_or_create_collection(
        name="financial_docs",
        embedding_function=emb_fn
    )

def extract_potential_doc_names(query: str, available_docs: list[str]) -> list[str]:
    # Very simple string match for doc names in the query
    matched = []
    query_lower = query.lower()
    for doc in available_docs:
        doc_clean = doc.replace("_", " ").replace("-", " ").lower()
        if doc_clean in query_lower:
            matched.append(doc)
            continue
            
        parts = [p for p in re.split(r'[_ -]', doc.lower()) if len(p) > 2]
        if parts and all(part in query_lower for part in parts if not part.isdigit()):
            # require year if present
            years = [p for p in parts if p.isdigit() and len(p) == 4]
            if years and years[0] not in query_lower:
                continue
            matched.append(doc)
            
    return matched

def retrieve_chunks(query: str, top_k: int = TOP_K_DEFAULT):
    collection = get_db_collection()
    
    # Get available doc names to apply the filtering trick
    existing_metadata = collection.get(include=["metadatas"])["metadatas"]
    available_docs = set()
    if existing_metadata:
        for m in existing_metadata:
            if m and "doc_name" in m:
                available_docs.add(m["doc_name"])
                
    where_clause = None
    if available_docs:
        matched_docs = extract_potential_doc_names(query, list(available_docs))
        if matched_docs:
            if len(matched_docs) == 1:
                where_clause = {"doc_name": matched_docs[0]}
            else:
                where_clause = {"doc_name": {"$in": matched_docs}}
                
    try:
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        print(f"Retrieval error with where_clause {where_clause}: {e}")
        results = {'ids': []}
        
    if not results.get('ids') or not results['ids'] or not results['ids'][0]:
        print(f"No results found with filtering {where_clause}. Falling back to unfiltered search.")
        # Fallback to unfiltered search if filtered search yields nothing
        if where_clause:
            results = collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
    retrieved = []
    if results.get('ids') and results['ids'] and results['ids'][0]:
        print(f"Found {len(results['ids'][0])} chunks.")
        for i in range(len(results['ids'][0])):
            distance = results['distances'][0][i]
            # Chroma cosine distance is 1 - cosine_similarity
            similarity = 1.0 - distance 
            
            meta = results['metadatas'][0][i]
            chunk_id = results['ids'][0][i]
            
            print(f"Retrieved Chunk {i+1}: ID={chunk_id}, Sim={similarity:.3f}, Doc={meta.get('doc_name')}")
            
            retrieved.append({
                "chunk_id": chunk_id,
                "text": results['documents'][0][i],
                "snippet": meta.get("text_snippet", ""),
                "doc_name": meta.get("doc_name", ""),
                "page": meta.get("page", 0),
                "similarity": similarity
            })
            
    return retrieved
