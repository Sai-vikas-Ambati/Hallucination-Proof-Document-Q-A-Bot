import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rag.generate import generate_answer
from rag.guardrail import REFUSAL_MESSAGE

# Sample evaluation dataset
EVAL_DATA = [
    # Assuming some financial PDFs might be uploaded, we create generic questions
    # that we expect to be unanswerable if the exact context is missing.
    {"question": "Who won the Super Bowl in 2023?", "expected_answerable": False},
    {"question": "What is the recipe for chocolate cake?", "expected_answerable": False},
    {"question": "What was the company's net income?", "expected_answerable": True},
    {"question": "How much did operating expenses increase?", "expected_answerable": True}
]

def run_evaluation():
    print("Running Evaluation...\n")
    correct_refusals = 0
    total_unanswerable = 0
    
    total_answerable = 0
    answered_with_citations = 0
    
    for item in EVAL_DATA:
        query = item["question"]
        expected = item["expected_answerable"]
        
        print(f"Q: {query}")
        result = generate_answer(query, top_k=5, sim_threshold=0.35)
        ans = result["answer"]
        cites = result["citations"]
        
        is_refusal = (ans == REFUSAL_MESSAGE)
        
        if not expected:
            total_unanswerable += 1
            if is_refusal:
                correct_refusals += 1
                print("  -> Correctly refused.")
            else:
                print(f"  -> Failed refusal! Answered: {ans}")
        else:
            total_answerable += 1
            if not is_refusal and len(cites) > 0:
                answered_with_citations += 1
                print(f"  -> Answered with {len(cites)} citations.")
            else:
                print("  -> Failed to answer or provide citations (might be correct if DB is empty).")
                
    print("\n--- Evaluation Results ---")
    if total_unanswerable > 0:
        print(f"Refusal Precision/Recall (Unanswerable): {correct_refusals}/{total_unanswerable} ({correct_refusals/total_unanswerable*100:.1f}%)")
    if total_answerable > 0:
        print(f"Answer Rate (Answerable, requires loaded PDFs): {answered_with_citations}/{total_answerable} ({answered_with_citations/total_answerable*100:.1f}%)")
        
if __name__ == "__main__":
    run_evaluation()
