import os
import sys
from dotenv import load_dotenv
from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore

load_dotenv()

DATA_DIR = "docs"
STORE_DIR = "faiss_store"

if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "What are the key features of the CPaaS platform?"

    # Ingest: load documents, chunk, embed, and persist the FAISS index
    docs = load_all_documents(DATA_DIR)
    store = FaissVectorStore(STORE_DIR)
    store.build_from_documents(docs)

    # Retrieval preview
    print(f"\nTop matches for: {query!r}")
    for r in store.query(query, top_k=3):
        print(f"- distance={r['distance']:.3f}: {r['metadata']['text'][:200]!r}")

    # Generation (needs an Anthropic API key)
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n[WARN] ANTHROPIC_API_KEY not set; skipping LLM answer. Add it to .env to enable.")
        sys.exit(0)

    from src.search import RAGSearch
    rag_search = RAGSearch(STORE_DIR, data_dir=DATA_DIR)
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("\nSummary:", summary)
