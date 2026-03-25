"""
Streamlit Frontend — Career Q&A
=================================

A clean web UI where anyone can ask questions about your professional
experience. Built on top of the Step 2 RAG chain.

Run locally:
    streamlit run app.py

Deploy:
    Push to GitHub, connect to Streamlit Community Cloud (free)
"""

import streamlit as st
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
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq")

# ── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Ask Krishna — Career Q&A",
    page_icon="🔍",
    layout="centered",
)

# ── Custom Styling ──────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem;
    }
    .main-header h1 {
        font-size: 2rem;
        margin-bottom: 0.25rem;
    }
    .main-header p {
        color: #888;
        font-size: 1rem;
    }
    .source-tag {
        display: inline-block;
        background: #f0f0f0;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        color: #666;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🔍 Ask Krishna</h1>
    <p>Ask anything about my professional experience — powered by RAG</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Load Resources (cached so they don't reload every interaction) ──────────

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    return vectorstore


@st.cache_resource
def get_llm():
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            st.error("GROQ_API_KEY not found in .env")
            st.stop()
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            api_key=api_key,
        )
    elif LLM_PROVIDER == "claude":
        from langchain_anthropic import ChatAnthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            st.error("ANTHROPIC_API_KEY not found in .env")
            st.stop()
        return ChatAnthropic(
            model="claude-sonnet-4-20250514",
            temperature=0.2,
            api_key=api_key,
        )


RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a career advisor assistant representing a product manager with 13+
years of experience across auto parts (NAPA/Equifax), sales tech (Salesloft),
and crypto loyalty (Bakkt). You also know about their portfolio projects
and education at Georgia Tech.

Answer the question based ONLY on the provided context. Be specific,
cite examples from the context, and be conversational but professional.

If the context doesn't contain enough information, say so honestly rather
than making things up.

Context:
{context}

Question: {question}

Answer:
""")


def format_docs(docs):
    formatted = []
    for doc in docs:
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        formatted.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


@st.cache_resource
def build_chain():
    vectorstore = load_vectorstore()
    llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, vectorstore


# ── Initialize ──────────────────────────────────────────────────────────────

chain, vectorstore = build_chain()
doc_count = vectorstore._collection.count()

st.caption(f"📚 Knowledge base: {doc_count} embedded chunks")

# ── Suggested Questions ─────────────────────────────────────────────────────

st.markdown("**Try asking:**")

col1, col2 = st.columns(2)

suggestions = [
    "What ML projects have you worked on?",
    "Tell me about a time you handled a data quality issue",
    "What's your experience with stakeholder management?",
    "How have you used AI in your product work?",
]

# Store which suggestion was clicked
if "selected_question" not in st.session_state:
    st.session_state.selected_question = None

with col1:
    if st.button(suggestions[0], use_container_width=True):
        st.session_state.selected_question = suggestions[0]
    if st.button(suggestions[1], use_container_width=True):
        st.session_state.selected_question = suggestions[1]

with col2:
    if st.button(suggestions[2], use_container_width=True):
        st.session_state.selected_question = suggestions[2]
    if st.button(suggestions[3], use_container_width=True):
        st.session_state.selected_question = suggestions[3]

st.divider()

# ── Chat Interface ──────────────────────────────────────────────────────────

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Determine input: either from suggestion button or typed input
user_input = st.chat_input("Ask about my experience...")

# If a suggestion was clicked, use that
if st.session_state.selected_question:
    user_input = st.session_state.selected_question
    st.session_state.selected_question = None

# Process input
if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching experience and generating answer..."):
            try:
                response = chain.invoke(user_input)
                st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
            except Exception as e:
                error_msg = f"Something went wrong: {str(e)}"
                st.error(error_msg)
                if "rate_limit" in str(e).lower():
                    st.info("Rate limit hit — wait a moment and try again.")

# ── Footer ──────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Built with LangChain + ChromaDB + Groq | "
    "[View source](https://github.com/krisdam) | "
    "RAG-powered career Q&A"
)
