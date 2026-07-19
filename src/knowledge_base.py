"""
knowledge_base.py
------------------
Handles all interaction with the ChromaDB vector database:
  - creating/loading a persistent collection
  - seeding it with a starter set of sports facts (data/sports_facts.json)
  - adding new facts (e.g. pulled from web search) so future quizzes stay fresh
  - semantic retrieval of the most relevant facts for a given sport/topic

ChromaDB's default embedding function (all-MiniLM-L6-v2, run locally via onnxruntime)
is used out of the box, so no extra embedding API/key is required. You can swap in
a different embedding function (e.g. OpenAI or Voyage embeddings) by passing
`embedding_function=` to chromadb's get_or_create_collection call below.
"""

import json
import os
from pathlib import Path
from typing import List, Dict

import chromadb

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_PATH = BASE_DIR / "chroma_store"
SEED_DATA_PATH = BASE_DIR / "data" / "sports_facts.json"
COLLECTION_NAME = "sports_knowledge"


class KnowledgeBase:
    def __init__(self, persist_path: str = str(CHROMA_PATH)):
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Sports facts used to ground quiz generation"},
        )

    def is_seeded(self) -> bool:
        return self.collection.count() > 0

    def seed_from_file(self, path: str = str(SEED_DATA_PATH)) -> int:
        """Load the starter JSON file into ChromaDB. Safe to call repeatedly
        (uses upsert, so re-running just refreshes the same IDs)."""
        with open(path, "r", encoding="utf-8") as f:
            facts = json.load(f)
        self.add_facts(facts)
        return len(facts)

    def add_facts(self, facts: List[Dict]) -> None:
        """
        facts: list of dicts like {"id": str, "sport": str, "text": str}
        Uses upsert so it's idempotent and safe to call for freshly
        web-searched content too.
        """
        if not facts:
            return
        self.collection.upsert(
            ids=[f["id"] for f in facts],
            documents=[f["text"] for f in facts],
            metadatas=[{"sport": f["sport"]} for f in facts],
        )

    def retrieve(self, sport: str, query: str = "", n_results: int = 6) -> List[str]:
        """
        Semantic retrieval of the most relevant stored facts for a sport.
        `query` lets you bias retrieval towards a sub-topic (e.g. "world cup winners"),
        otherwise it defaults to just searching on the sport name.
        """
        search_text = query.strip() if query.strip() else sport
        results = self.collection.query(
            query_texts=[search_text],
            n_results=n_results,
            where={"sport": sport},
        )
        docs = results.get("documents", [[]])[0]
        return docs


def get_knowledge_base(auto_seed: bool = True) -> KnowledgeBase:
    """Convenience factory: returns a ready-to-use KnowledgeBase, seeding it
    on first run if it's empty."""
    kb = KnowledgeBase()
    if auto_seed and not kb.is_seeded():
        kb.seed_from_file()
    return kb


if __name__ == "__main__":
    # Quick manual test: `python src/knowledge_base.py`
    kb = get_knowledge_base()
    print(f"Collection has {kb.collection.count()} documents")
    print(kb.retrieve("Badminton", "Thomas Cup"))
