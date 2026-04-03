"""
Step 1: Chunk & Embed Career Documents into ChromaDB
=====================================================

What you'll learn:
- How document ingestion works (loading PDFs, markdown, text)
- Chunking strategies (why size and overlap matter)
- How embeddings turn text into searchable vectors
- How a vector store (ChromaDB) indexes and retrieves those vectors

What you'll have at the end:
- Your career docs embedded in a local ChromaDB instance
- A working semantic search: "data-driven decision making" → returns your NAPA story
"""

# ── 1. Imports ──────────────────────────────────────────────────────────────

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    PyMuPDFLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

# ── 2. Configuration ───────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CHROMA_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Chunking parameters — tune these as you learn
CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 100    # overlap between chunks (preserves context at boundaries)


# ── 3. Load Documents ──────────────────────────────────────────────────────

def load_documents(data_dir: str):
    """
    Load all documents from the data directory.

    WHY THIS MATTERS:
    Different file types need different loaders. PDFs have embedded fonts
    and layouts, markdown has headers and formatting, text is plain.
    Each loader extracts the raw text content so the rest of the pipeline
    can work with it uniformly.
    """
    documents = []

    # Load markdown files (.md)
    md_loader = DirectoryLoader(
        data_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        show_progress=True,
    )
    documents.extend(md_loader.load())

    # Load PDFs
    pdf_loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=True,
    )
    documents.extend(pdf_loader.load())

    # Load plain text
    txt_loader = DirectoryLoader(
        data_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
    )
    documents.extend(txt_loader.load())

    print(f"✅ Loaded {len(documents)} documents from {data_dir}")
    return documents


# ── 4. Chunk Documents ─────────────────────────────────────────────────────

def chunk_documents(documents, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split documents into smaller chunks for embedding.

    WHY THIS MATTERS:
    LLMs have context windows, and embeddings work best on focused passages.
    If you embed an entire 5-page resume as one vector, the embedding is a
    blurry average of everything. Chunking creates focused, retrievable units.

    RecursiveCharacterTextSplitter tries to split on natural boundaries:
    paragraphs → sentences → words, keeping chunks under the size limit.

    CHUNK_SIZE and CHUNK_OVERLAP are your main tuning knobs:
    - Bigger chunks = more context per retrieval, but less precise matching
    - More overlap = better continuity at boundaries, but more redundancy
    - Start with 500/100 and adjust based on retrieval quality
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],  # Split on headers first
    )

    chunks = splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")

    # Preview first few chunks so you can see what's happening
    print("\n── Sample Chunks ──")
    for i, chunk in enumerate(chunks[:3]):
        source = chunk.metadata.get("source", "unknown")
        print(f"\nChunk {i+1} (from {os.path.basename(source)}):")
        print(f"  Length: {len(chunk.page_content)} chars")
        print(f"  Preview: {chunk.page_content[:150]}...")

    return chunks


# ── 5. Embed & Store ───────────────────────────────────────────────────────

def embed_and_store(chunks, persist_dir=CHROMA_DIR, model_name=EMBEDDING_MODEL):
    """
    Convert chunks into vectors and store in ChromaDB.

    WHY THIS MATTERS:
    This is the core of RAG. Each chunk gets converted into a high-dimensional
    vector (384 dimensions for MiniLM). Similar text → similar vectors.

    When you later ask "what's my best example of stakeholder management?",
    that question gets embedded into the same vector space, and ChromaDB
    finds the chunks with the closest vectors — your NAPA taxonomy story
    about managing merchandising team opinions.

    ChromaDB persists to disk so you only need to run ingestion once
    (unless you update your documents).
    """
    print(f"\n⏳ Loading embedding model: {model_name}")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},  # Use "cuda" if you have a GPU
    )

    print(f"⏳ Embedding {len(chunks)} chunks and storing in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    print(f"✅ Vector store created at {persist_dir}")
    print(f"   Collection size: {vectorstore._collection.count()} vectors")

    return vectorstore


# ── 6. Test Retrieval ──────────────────────────────────────────────────────

def test_retrieval(vectorstore, queries=None):
    """
    Run sample queries to verify the system works.

    This is your sanity check. If "data-driven decision" doesn't return
    your NAPA interchange story, something's off with chunking or embedding.
    """
    if queries is None:
        queries = [
            "data-driven decision making",
            "ML classification project",
            "stakeholder management",
            "rollback or recovery from a mistake",
            "agentic AI or prompt chaining",
        ]

    print("\n── Test Retrieval ──")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        results = retriever.invoke(query)
        for i, doc in enumerate(results):
            source = os.path.basename(doc.metadata.get("source", "unknown"))
            print(f"   Result {i+1} [{source}]: {doc.page_content[:120]}...")


# ── 7. Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Step 1: Career Document Ingestion Pipeline")
    print("=" * 60)

    # Load → Chunk → Embed → Test
    documents = load_documents(DATA_DIR)

    if not documents:
        print("\n⚠️  No documents found! Add your files to project1_rag/data/")
        print("   Supported formats: .md, .pdf, .txt")
        print("   Start with star_stories.md — it's already there as a template.")
        exit(1)

    chunks = chunk_documents(documents)
    vectorstore = embed_and_store(chunks)
    test_retrieval(vectorstore)

    print("\n" + "=" * 60)
    print("✅ Step 1 complete! Your career docs are now searchable.")
    print("   Next: Step 2 — Build a retrieval chain with an LLM.")
    print("=" * 60)
