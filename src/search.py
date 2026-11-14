import os
import anthropic
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = None, data_dir: str = "docs"):
        self.llm_model = llm_model or os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5")
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.data_loader import load_all_documents
            docs = load_all_documents(data_dir)
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        # Reads ANTHROPIC_API_KEY from the environment / .env
        self.client = anthropic.Anthropic()
        print(f"[INFO] Anthropic client initialized: {self.llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        response = self.client.beta.messages.create(
            model=self.llm_model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system="Answer the user's question using only the provided context. If the context doesn't contain the answer, say so.",
            messages=[{
                "role": "user",
                "content": f"<context>\n{context}\n</context>\n\nQuestion: {query}",
            }],
        )
        if response.stop_reason == "refusal":
            return "The model declined to answer this request."
        return "".join(b.text for b in response.content if b.type == "text")

# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What are the key features of the CPaaS platform?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
