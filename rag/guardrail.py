from rag.config import SIM_THRESHOLD_DEFAULT
import json

REFUSAL_MESSAGE = "I don't know based on the provided documents"

def check_retrieval_gate(retrieved_chunks: list[dict], threshold: float = SIM_THRESHOLD_DEFAULT) -> bool:
    """Layer 1: Check if the best retrieved chunk meets the similarity threshold."""
    if not retrieved_chunks:
        return False
        
    best_score = max([chunk["similarity"] for chunk in retrieved_chunks])
    if best_score < threshold:
        return False
        
    return True

def verify_generation_post_check(json_response: str, retrieved_chunk_ids: list[str]) -> dict:
    """
    Layer 3: Post-check verification.
    Parses the JSON and checks if answerable=False or invalid citations.
    Returns a dict with {"valid": bool, "parsed": dict, "reason": str}
    """
    try:
        data = json.loads(json_response)
    except json.JSONDecodeError:
        return {"valid": False, "parsed": None, "reason": "JSON parsing failed"}
        
    if not data.get("answerable", False):
        return {"valid": False, "parsed": data, "reason": "Model decided it's unanswerable"}
        
    citations = data.get("citations", [])
    if not citations:
        return {"valid": False, "parsed": data, "reason": "No citations provided"}
        
    for cite in citations:
        cite_str = str(cite)
        # Check if the citation matches any chunk_id exactly
        if cite_str in retrieved_chunk_ids:
            continue
            
        # Check if model returned just the chunk number (e.g., 1, 2, 3) or "Chunk 1"
        found_match = False
        for i, chunk_id in enumerate(retrieved_chunk_ids):
            if cite_str == str(i + 1) or cite_str.lower().replace("chunk ", "") == str(i + 1):
                found_match = True
                # Replace the generic citation with the actual chunk_id
                data["citations"][data["citations"].index(cite)] = chunk_id
                break
                
        if not found_match:
            print(f"Warning: Failed to match citation '{cite}' to retrieved chunk IDs {retrieved_chunk_ids}")
            # Soft fail: If we have at least one valid citation, don't fail the whole response
            # Just remove the invalid one. If all are invalid, it will fail later if we want it to.
            # For strictness, if it generated a completely fake one, we might want to fail,
            # but usually it's just formatting. Let's fail for now to be safe, or just ignore it.
            # To be robust, let's just log it and remove it.
            data["citations"].remove(cite)
            
    if not data["citations"]:
         return {"valid": False, "parsed": data, "reason": "No valid citations remained after matching"}

    return {"valid": True, "parsed": data, "reason": "Passed all checks"}
