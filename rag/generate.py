from groq import Groq
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rag.config import GROQ_API_KEY, GROQ_MODEL, SIM_THRESHOLD_DEFAULT, TOP_K_DEFAULT
from rag.retrieve import retrieve_chunks
from rag.guardrail import check_retrieval_gate, verify_generation_post_check, REFUSAL_MESSAGE
import json

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

SYSTEM_PROMPT = """You are a strict, hallucination-proof financial Q&A bot.
You will be provided with context chunks from financial documents. Each chunk is prefixed with its source and accompanied by a chunk_id.
Your task is to answer the user's question ONLY using the provided context.

Rules:
1. Answer ONLY from the context. Never use outside knowledge.
2. Quote numbers exactly with units and fiscal period.
3. If arithmetic is needed, show it using only numbers present in the context.
4. If the context does not contain the answer, you MUST set "answerable" to false.
5. Provide your response in valid JSON format.

Required JSON format:
{
  "answerable": true or false,
  "answer": "your detailed answer here, or empty if unanswerable",
  "citations": ["chunk_id_1", "chunk_id_2"]
}
"""

def generate_answer(query: str, top_k: int = TOP_K_DEFAULT, sim_threshold: float = SIM_THRESHOLD_DEFAULT):
    start_time = time.time()
    
    if not client:
        return {
            "answer": "Error: GROQ_API_KEY not found in .env.",
            "citations": [],
            "latency": 0
        }

    # 1. Retrieve
    chunks = retrieve_chunks(query, top_k=top_k)
    
    # 2. Guardrail Layer 1: Retrieval Gate
    if not check_retrieval_gate(chunks, threshold=sim_threshold):
        latency = time.time() - start_time
        return {
            "answer": REFUSAL_MESSAGE,
            "citations": [],
            "latency": latency
        }
        
    # Prepare context
    context_text = ""
    valid_chunk_ids = []
    chunk_map = {}
    
    for i, chunk in enumerate(chunks):
        cid = chunk["chunk_id"]
        valid_chunk_ids.append(cid)
        chunk_map[cid] = chunk
        context_text += f"\n--- Chunk {i+1} (ID: {cid}) ---\n{chunk['text']}\n"
        
    prompt = f"Context:\n{context_text}\n\nQuestion: {query}"
    
    # 3. Guardrail Layer 2 & Generate
    def call_llm():
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
        
    raw_response = call_llm()
    print(f"\nRAW LLM RESPONSE:\n{raw_response}\n")
    
    # 4. Guardrail Layer 3: Post-check
    check_result = verify_generation_post_check(raw_response, valid_chunk_ids)
    
    if not check_result["valid"] and check_result["reason"] == "JSON parsing failed":
        # Retry once
        raw_response = call_llm()
        check_result = verify_generation_post_check(raw_response, valid_chunk_ids)
        
    latency = time.time() - start_time

    if not check_result["valid"]:
        return {
            "answer": REFUSAL_MESSAGE,
            "citations": [],
            "latency": latency
        }
        
    parsed = check_result["parsed"]
    
    # Map citations to full objects
    final_citations = []
    for cid in parsed.get("citations", []):
        if cid in chunk_map:
            c = chunk_map[cid]
            final_citations.append({
                "doc_name": c["doc_name"],
                "page": c["page"],
                "snippet": c["snippet"],
                "similarity": c["similarity"]
            })
            
    return {
        "answer": parsed["answer"],
        "citations": final_citations,
        "latency": latency
    }
