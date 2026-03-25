"""
Step 2: Build a Retrieval Chain (RAG Q&A)
==========================================

What you'll learn:
- How to connect a vector store to an LLM
- How the RAG pattern works: retrieve context → inject into prompt → generate answer
- How to use Groq (free cloud API) as the LLM
- How to swap to Claude API when you're ready to demo

What you'll have at the end:
- A working Q&A system: ask a question about your career, get an answer with sources

Prerequisites:
- Step 1 complete (documents embedded in ChromaDB)
- Groq API key from console.groq.com (free signup)
"""

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────

CHROMA_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# LLM provider toggle — change this to switch between Groq (free) and Claude (paid)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq")  # "groq" or "claude"


# ── 1. Load the Vector Store ───────────────────────────────────────────────

def load_vectorstore():
    """Load the persisted ChromaDB from Step 1."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    print(f"✅ Loaded vector store ({vectorstore._collection.count()} vectors)")
    return vectorstore


# ── 2. Set Up the LLM ──────────────────────────────────────────────────────

def get_llm(provider=LLM_PROVIDER):
    """
    Initialize the LLM based on the provider setting.

    HOW THIS WORKS:
    Both Groq and Claude follow the same LangChain interface (BaseChatModel).
    That means the rest of your pipeline — retrieval, prompting, parsing —
    doesn't care which one you're using. You swap the brain, everything
    else stays identical.

    GROQ (default, free):
    - Runs Llama 3.1 70B on Groq's cloud hardware
    - Free tier: 30 requests/minute, 15,000 tokens/minute
    - More than enough for development and testing
    - Sign up at console.groq.com, grab an API key, done

    CLAUDE (for demo/portfolio):
    - Higher quality outputs, especially for nuanced answers
    - Costs money per token, but minimal for a portfolio demo
    - Swap to this when you're recording a demo or showing to recruiters
    """
    if provider == "groq":
        from langchain_groq import ChatGroq

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("❌ GROQ_API_KEY not found in .env")
            print("   1. Sign up at https://console.groq.com (free)")
            print("   2. Create an API key")
            print("   3. Add GROQ_API_KEY=gsk_your-key-here to your .env file")
            exit(1)

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            api_key=api_key,
        )
        print("✅ LLM ready: Llama 3.3 70B (via Groq, free)")

    elif provider == "claude":
        from langchain_anthropic import ChatAnthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("❌ ANTHROPIC_API_KEY not found in .env")
            print("   Add ANTHROPIC_API_KEY=sk-ant-... to your .env file")
            exit(1)

        llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            temperature=0.2,
            api_key=api_key,
        )
        print("✅ LLM ready: Claude Sonnet (via Anthropic API, paid)")

    else:
        print(f"❌ Unknown LLM_PROVIDER: {provider}")
        print("   Set LLM_PROVIDER=groq or LLM_PROVIDER=claude in .env")
        exit(1)

    return llm


# ── 3. Build the RAG Chain ─────────────────────────────────────────────────

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a career advisor assistant with deep knowledge of the user's
professional experience. Answer the question based ONLY on the provided context.

If the context doesn't contain enough information to answer, say so honestly.
Always cite which source document your answer came from.

Context:
{context}

Question: {question}

Answer (cite your sources):
""")


def format_docs(docs):
    """Format retrieved documents into a single context string."""
    formatted = []
    for i, doc in enumerate(docs):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        formatted.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_rag_chain(vectorstore, llm):
    """
    Build the full RAG chain: question → retrieve → format → prompt → LLM → answer

    WHY THIS MATTERS:
    This is the core RAG pattern you'll see everywhere:
    1. User asks a question
    2. The question gets embedded and used to search the vector store
    3. Top-k relevant chunks are retrieved
    4. Chunks are injected into the prompt as "context"
    5. The LLM generates an answer grounded in that context

    The chain is composable — each piece is a separate "runnable" that
    can be swapped, logged, or modified independently.
    """
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    print("✅ RAG chain built")
    return chain


# ── 4. Interactive Q&A Loop ────────────────────────────────────────────────

def run_qa_loop(chain):
    """Simple terminal-based Q&A for testing."""
    print("\n" + "=" * 60)
    print("Career Q&A — Ask about your experience")
    print(f"Using: {LLM_PROVIDER.upper()}")
    print("Type 'quit' to exit")
    print("=" * 60)

    sample_questions = [
        "What's my best example of data-driven decision making?",
        "Tell me about a time I had to manage a rollback or failure.",
        "What ML projects have I worked on?",
        "How have I demonstrated stakeholder management?",
        "What do I know about agentic AI vs prompt chaining?",
    ]

    print("\n💡 Try these sample questions:")
    for i, q in enumerate(sample_questions, 1):
        print(f"   {i}. {q}")

    while True:
        print()
        question = input("🔍 Your question: ").strip()

        if question.lower() in ("quit", "exit", "q"):
            print("👋 Done!")
            break

        if not question:
            continue

        # Handle numbered shortcuts
        if question.isdigit() and 1 <= int(question) <= len(sample_questions):
            question = sample_questions[int(question) - 1]
            print(f"   → {question}")

        print("\n⏳ Thinking...")
        try:
            answer = chain.invoke(question)
            print(f"\n📝 Answer:\n{answer}")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            if "rate_limit" in str(e).lower():
                print("   Groq free tier rate limit hit. Wait a moment and try again.")


# ── 5. Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Step 2: Career RAG Q&A")
    print("=" * 60)

    vectorstore = load_vectorstore()
    llm = get_llm()
    chain = build_rag_chain(vectorstore, llm)
    run_qa_loop(chain)
