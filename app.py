import streamlit as st
import os
from pathlib import Path
from rag.config import PDF_DIR, SIM_THRESHOLD_DEFAULT, TOP_K_DEFAULT, GROQ_API_KEY
from rag.generate import generate_answer
from rag.ingest import ingest_pdfs
from rag.guardrail import REFUSAL_MESSAGE

st.set_page_config(page_title="Hallucination-Proof Q&A", page_icon="🏦", layout="wide")

os.makedirs(PDF_DIR, exist_ok=True)

st.title("Hallucination-Proof Document Q&A Bot")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing. Please add it to your .env file.")
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.header("Architecture & Config")
    st.markdown("""
    **Pipeline**:
    1. **Ingest**: PyMuPDF -> BAAI/bge-small-en-v1.5 -> ChromaDB. Chunks are strictly page-bounded.
    2. **Retrieve**: Cosine similarity. Uses company name metadata filtering if mentioned in the query.
    3. **Guardrails**: 
       - Similarity threshold gate.
       - Strict prompt enforcing valid JSON.
       - Post-generation citation validation.
    """)
    
    st.subheader("Settings")
    top_k = st.slider("Top-K Chunks", 1, 10, TOP_K_DEFAULT)
    sim_threshold = st.slider("Similarity Threshold", 0.0, 1.0, SIM_THRESHOLD_DEFAULT)
    
    st.subheader("Upload PDF")
    uploaded_file = st.file_uploader("Upload a financial PDF", type=["pdf"])
    if uploaded_file:
        save_path = Path(PDF_DIR) / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.spinner(f"Indexing {uploaded_file.name}..."):
            ingest_pdfs(PDF_DIR)
        st.success(f"Indexed {uploaded_file.name} successfully!")

    st.subheader("Indexed Documents")
    try:
        from rag.retrieve import get_db_collection
        coll = get_db_collection()
        docs = set()
        meta = coll.get(include=["metadatas"])["metadatas"]
        if meta:
            for m in meta:
                if m and "doc_name" in m:
                    docs.add(m["doc_name"])
        if docs:
            for d in docs:
                st.write(f"📄 {d}")
        else:
            st.write("No documents indexed yet.")
    except Exception as e:
        st.write("Could not load index.")

    st.subheader("Sample Questions")
    sample_q1 = "What is the total revenue?"
    sample_q2 = "Who won the Super Bowl?"
    if st.button(sample_q1):
        st.session_state["sample_query"] = sample_q1
    if st.button(sample_q2):
        st.session_state["sample_query"] = sample_q2

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg["content"] == REFUSAL_MESSAGE:
            st.warning(msg["content"])
        else:
            st.markdown(msg["content"])
            
        if "citations" in msg and msg["citations"]:
            for i, cite in enumerate(msg["citations"]):
                with st.expander(f"Source {i+1}: {cite['doc_name']} (Page {cite['page']}) - Sim: {cite['similarity']:.2f}"):
                    st.markdown(f"_{cite['snippet']}_")
        
        if "latency" in msg:
            st.caption(f"Latency: {msg['latency']:.2f}s")

# Handle input
user_query = st.chat_input("Ask a question about the financial documents...")

# Check if a sample query was clicked
if "sample_query" in st.session_state and st.session_state["sample_query"]:
    user_query = st.session_state["sample_query"]
    st.session_state["sample_query"] = None

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents..."):
            try:
                result = generate_answer(user_query, top_k=top_k, sim_threshold=sim_threshold)
                ans = result["answer"]
                citations = result["citations"]
                latency = result.get("latency", 0)
                
                if ans == REFUSAL_MESSAGE:
                    st.warning(ans)
                else:
                    st.markdown(ans)
                    
                if citations:
                    for i, cite in enumerate(citations):
                        with st.expander(f"Source {i+1}: {cite['doc_name']} (Page {cite['page']}) - Sim: {cite['similarity']:.2f}"):
                            st.markdown(f"_{cite['snippet']}_")
                            
                st.caption(f"Latency: {latency:.2f}s")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": ans,
                    "citations": citations,
                    "latency": latency
                })
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
