# AI-Powered Sports Quiz Generation Agent

An AI agent that generates factually-grounded, multiple-choice sports quizzes
for social media, using Retrieval-Augmented Generation (RAG): a ChromaDB
vector database of sports facts, live web search for freshness, and Claude
as the generation model — wrapped in a Streamlit dashboard.

## How it works (architecture)

```
┌─────────────────┐
│  Streamlit UI    │  user picks sport + difficulty, clicks Generate/Regenerate
│    (app.py)      │
└────────┬─────────┘
         │
         v
┌─────────────────────┐
│    QuizAgent          │  src/quiz_agent.py
│                        │
│  1. picks a random     │
│     "topic angle"      │  (e.g. "record holders", "iconic matches")
│     so regenerated      │  quizzes don't repeat themselves
│     quizzes differ      │
│                        │
│  2. retrieves grounded │──> ChromaDB (src/knowledge_base.py)
│     facts for the      │    persistent vector store, seeded from
│     sport + angle      │    data/sports_facts.json, semantic search
│                        │    via sentence-transformer embeddings
│  3. optionally runs a  │──> Web Search (src/web_search.py)
│     live web search    │    duckduckgo-search, no API key needed
│     for recent info,    │    results are also written back into
│     and folds results   │    ChromaDB so the KB gets fresher over time
│     into the context    │
│                        │
│  4. sends context +    │──> Groq (free, no card required)
│     instructions to     │    system prompt forces "only use supplied
│     the LLM, gets       │    context" grounding to reduce hallucination
│     back strict JSON    │
└─────────────────────┘
         │
         v
   Structured quiz (sport, difficulty, questions[], options, correct
   answer, explanation) rendered in the dashboard
```

### Anti-hallucination strategy
- The system prompt explicitly forbids inventing facts not present in the
  retrieved CONTEXT, and instructs the model to pick a different, supported
  angle rather than guess.
- All generated questions must be traceable to either the ChromaDB facts or
  the live web search snippets injected into the prompt.
- Web search results are persisted back into ChromaDB, so the knowledge base
  becomes a growing, reusable source of truth rather than being thrown away
  after each request.

### Freshness / diversity strategy
- Each generation call randomly selects a "topic angle" (tournament winners,
  records, rules, iconic moments, milestones, team history) to bias retrieval
  and steer the LLM, so **Regenerate** produces genuinely different questions
  instead of near-duplicates.
- Live web search (toggle in the sidebar) pulls in information that may be
  more recent than the seeded knowledge base.

## Project structure

```
sports-quiz-agent/
├── app.py                  # Streamlit dashboard (entry point)
├── requirements.txt
├── .env.example
├── data/
│   └── sports_facts.json   # starter knowledge base (5 sports x 5 facts)
├── chroma_store/           # ChromaDB persistent storage (auto-created)
└── src/
    ├── knowledge_base.py   # ChromaDB setup, seeding, retrieval
    ├── web_search.py       # DuckDuckGo web search wrapper
    └── quiz_agent.py       # RAG orchestration + LLM call + JSON parsing
```

## Setup

1. **Clone / copy the project**, then create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your API key:**
   ```bash
   cp .env.example .env
   # then edit .env and paste your Groq API key
   ```
   Get a free key (no card required) from https://console.groq.com/keys

4. **Run the dashboard:**
   ```bash
   streamlit run app.py
   ```
   The first run will automatically seed ChromaDB from `data/sports_facts.json`
   (creates a `chroma_store/` folder for persistence).

5. **Use it:** pick a sport and difficulty in the sidebar, click **Generate
   Quiz**. Click **Regenerate** for a fresh set on the same settings.

## Extending the knowledge base

Add more facts to `data/sports_facts.json` (same `{id, sport, text}` shape)
and re-run the app — `KnowledgeBase.seed_from_file()` uses `upsert`, so it's
safe to re-seed. You can also call `KnowledgeBase.add_facts()` programmatically
to ingest facts from other sources (APIs, scrapers, etc.).

## Swapping components

- **LLM provider:** change the `model=` and client setup in `src/quiz_agent.py`.
- **Web search provider:** replace the implementation of `search_web()` in
  `src/web_search.py` with SerpAPI/Tavily/Bing — keep the same return shape.
- **Embeddings:** pass a custom `embedding_function` to
  `get_or_create_collection()` in `src/knowledge_base.py` if you want to use
  OpenAI/Voyage/Cohere embeddings instead of the default local model.

## Known limitations / next steps

- Web search quality depends on DuckDuckGo's free-tier results; a paid search
  API will give more reliable recency for fast-moving events.
- No persistence of generated quizzes/history yet — could add a "past quizzes"
  view backed by a small SQLite table.
- No answer-shuffling randomization guard yet (the correct answer's position
  is left to the model) — worth adding a post-processing shuffle for social
  media posts where option order matters.
- No automated fact-checking pass; for production use, consider a second LLM
  call that cross-checks each question against the context before publishing.
