# 🚀 Getting Started — Step 1

## What You're About To Do

1. Install dependencies
2. Add your career documents
3. Run the ingestion pipeline (chunk → embed → store)
4. Test semantic search on your own experience

Total time: ~30 minutes

---

## Prerequisites

### Install Ollama (free local LLM)
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows — download from https://ollama.ai
```

Then pull the model:
```bash
ollama pull llama3
```

### Set Up Python Environment
```bash
cd job-search-agent
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Create Your .env
```bash
cp .env.example .env
# No changes needed for Step 1 — defaults are fine
```

---

## Add Your Documents

The sample `star_stories.md` is already in `project1_rag/data/` with your
NAPA and MeetingIQ stories pre-filled as templates. To get the most out of
this step:

1. **Fill in the Salesloft story** in `star_stories.md`
2. **Add your resume** as `resume.pdf` in the data folder
3. **Optionally add** any other career docs: project writeups, interview
   notes, job criteria, etc.

The more documents you add, the richer the retrieval results.

---

## Run Ingestion

```bash
cd project1_rag
python src/ingest.py
```

You should see:
```
✅ Loaded 3 documents from data/
✅ Created 24 chunks (size=500, overlap=100)
⏳ Loading embedding model: all-MiniLM-L6-v2
⏳ Embedding 24 chunks and storing in ChromaDB...
✅ Vector store created at ./chroma_db

── Test Retrieval ──
🔍 Query: 'data-driven decision making'
   Result 1 [star_stories.md]: NAPA Interchange — Data Quality at Scale...
```

If the test queries return relevant chunks from your stories, you're good.
If not, try adjusting CHUNK_SIZE in `ingest.py` (larger = more context per chunk).

---

## What Just Happened (Concepts You Learned)

| Concept | What it means | Where it happened |
|---|---|---|
| **Document Loading** | Different file formats need different parsers | `load_documents()` |
| **Chunking** | Splitting docs into focused, searchable units | `chunk_documents()` |
| **Embeddings** | Converting text to numerical vectors for similarity search | `embed_and_store()` |
| **Vector Store** | A database optimized for finding similar vectors | ChromaDB in `embed_and_store()` |
| **Semantic Search** | Finding text by meaning, not keywords | `test_retrieval()` |

---

## Next Step

Once ingestion works, move to Step 2:
```bash
python src/rag_chain.py
```

This connects your vector store to an LLM so you can ask natural language
questions and get grounded answers. See `rag_chain.py` for details.
