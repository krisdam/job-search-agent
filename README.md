# 🔍 JobAgent — AI-Powered Job Search Pipeline

## The Problem
Job searching as a PM is a mess. You're juggling dozens of postings, tailoring your story for each, tracking applications, and matching experience to requirements — all manually.

**JobAgent** solves this with two layered AI systems:

---

## Project 1: Career RAG Q&A (`project1_rag/`)
> *"What's my best example of data-driven decision making?"*

A RAG (Retrieval-Augmented Generation) system that ingests your resume, STAR stories, project write-ups, and interview notes — then answers questions about your experience with source-cited responses.

**Demonstrates:** RAG pipeline, agent tool use, vector search, document parsing

### Tech Stack
| Technology | Purpose |
|---|---|
| LangChain | LLM orchestration framework |
| Ollama + Llama 3 | Free local LLM (swap to Claude for demo) |
| ChromaDB | Vector database for semantic search |
| sentence-transformers | Embedding model (`all-MiniLM-L6-v2`) |
| PyMuPDF / Unstructured | Document ingestion (PDF, DOCX, TXT) |
| Streamlit | Web UI |

---

## Project 2: Multi-Agent Job Search Pipeline (`project2_multiagent/`)
> *"Find AI PM roles, match them to my experience, and draft talking points."*

A multi-agent system where specialized agents collaborate via an orchestration graph:

| Agent | Role |
|---|---|
| **Scout** | Searches job boards, filters by criteria |
| **Match** | Scores postings against your RAG knowledge base |
| **Writer** | Drafts tailored talking points per role |
| **Orchestrator** | Manages state, routing, and handoffs |

**Demonstrates:** Multi-agent orchestration, persistent memory, tool use, live data integration

### Tech Stack (adds to Project 1)
| Technology | Purpose |
|---|---|
| LangGraph | Agent orchestration + state management |
| LangGraph Checkpointing | Persistent memory across sessions |
| SerpAPI / Apify | Live job posting search |
| Claude API | High-quality LLM for final demo |

---

## Build Sequence

Each step produces a working increment:

```
Step 1: Chunk & embed career docs into ChromaDB          → embeddings, vector stores
Step 2: Build retrieval chain ("what's my best X?")      → RAG fundamentals
Step 3: Add agent tools (keyword extractor, scorer)      → agent tool use
────────────────────────────────────────────────────────── Project 1 complete ──
Step 4: Add Match Agent (score postings vs experience)   → multi-agent handoff
Step 5: Add LangGraph orchestration + memory             → state management
Step 6: Add Scout Agent (live job search)                → API integration
Step 7: Full pipeline + Streamlit UI                     → end-to-end deployment
────────────────────────────────────────────────────────── Project 2 complete ──
```

---

## Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai) installed locally

### Install
```bash
# Clone the repo
git clone <your-repo-url>
cd job-search-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Pull the local LLM
ollama pull llama3

# Pull the embedding model (optional — sentence-transformers handles this)
ollama pull nomic-embed-text
```

### Run
```bash
# Step 1: Ingest your documents
python project1_rag/src/ingest.py

# Step 2+: Launch the Q&A interface
streamlit run frontend/app.py
```

---

## Data (Your Career Docs)

Place your files in `project1_rag/data/`:
```
data/
├── resume.pdf
├── star_stories.md        # Your STAR-format interview stories
├── project_writeups.md    # Portfolio project descriptions
├── interview_notes.md     # Past interview Qs and your answers
└── job_criteria.md        # What you're looking for in a role
```

---

## License
MIT
