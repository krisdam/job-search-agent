"""
Streamlit Frontend — Career Q&A (numpy vector search, no FAISS/ChromaDB)
"""
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import numpy as np
import os, pickle

load_dotenv()

FAISS_DIR = os.path.join(os.path.dirname(__file__), "faiss_index")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

st.set_page_config(page_title="Ask KD — Career Q&A", page_icon="🔍", layout="centered")

st.markdown("""
<style>
    .stApp { max-width: 800px; margin: 0 auto; }
    .main-header { text-align: center; padding: 1rem 0 0.5rem; }
    .main-header h1 { font-size: 2rem; margin-bottom: 0.25rem; }
    .main-header p { color: #888; font-size: 1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🔍 Ask KD</h1>
    <p>Ask anything about my professional experience — powered by RAG</p>
</div>
""", unsafe_allow_html=True)

st.divider()

@st.cache_resource
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_resource
def load_index():
    pkl_path = os.path.join(FAISS_DIR, "index.pkl")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    # data is a dict with docstore and index_to_docstore_id from FAISS
    # Instead load our custom numpy index if present, else fall back
    numpy_path = os.path.join(FAISS_DIR, "numpy_index.pkl")
    if os.path.exists(numpy_path):
        with open(numpy_path, "rb") as f:
            return pickle.load(f)
    return None

@st.cache_resource
def build_numpy_store():
    """Load docs from data folder and build in-memory numpy index."""
    import glob
    embedder = load_embedder()
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    
    texts = []
    sources = []
    
    # Load markdown files
    for path in glob.glob(os.path.join(data_dir, "*.md")):
        with open(path, "r") as f:
            content = f.read()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(content)
        for chunk in chunks:
            texts.append(chunk)
            sources.append(os.path.basename(path))
    
    # Load PDFs
    for path in glob.glob(os.path.join(data_dir, "*.pdf")):
        try:
            import fitz
            doc = fitz.open(path)
            content = ""
            for page in doc:
                content += page.get_text()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_text(content)
            for chunk in chunks:
                texts.append(chunk)
                sources.append(os.path.basename(path))
        except Exception as e:
            st.warning(f"Could not load {path}: {e}")
    
    if not texts:
        st.error("No documents found in data folder!")
        st.stop()
    
    embeddings = embedder.encode(texts, show_progress_bar=False)
    return {"texts": texts, "sources": sources, "embeddings": embeddings}

def similarity_search(query, store, embedder, k=4):
    query_vec = embedder.encode([query])[0]
    embeddings = store["embeddings"]
    scores = np.dot(embeddings, query_vec) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec) + 1e-10
    )
    top_k = np.argsort(scores)[::-1][:k]
    results = []
    for i in top_k:
        results.append({"text": store["texts"][i], "source": store["sources"][i]})
    return results

@st.cache_resource
def get_llm():
    from langchain_groq import ChatGroq
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found")
        st.stop()
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2, api_key=api_key)

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a career advisor assistant representing a product manager with 13+
years of experience across auto parts (NAPA/Equifax), sales tech (Salesloft),
and crypto loyalty (Bakkt). You also know about their portfolio projects
and education at Georgia Tech.

Answer the question based ONLY on the provided context. Be specific,
cite examples from the context, and be conversational but professional.

If the context does not contain enough information, say so honestly.

Context:
{context}

Question: {question}

Answer:
""")

embedder = load_embedder()
store = build_numpy_store()
llm = get_llm()
chain = RAG_PROMPT | llm | StrOutputParser()

st.caption(f"📚 Knowledge base: {len(store['texts'])} embedded chunks")

st.markdown("**Try asking:**")
col1, col2 = st.columns(2)
suggestions = [
    "What ML projects have you worked on?",
    "Tell me about a time you handled a data quality issue",
    "What is your experience with stakeholder management?",
    "How have you used AI in your product work?",
]

if "selected_question" not in st.session_state:
    st.session_state.selected_question = None

with col1:
    if st.button(suggestions[0], use_container_width=True): st.session_state.selected_question = suggestions[0]
    if st.button(suggestions[1], use_container_width=True): st.session_state.selected_question = suggestions[1]
with col2:
    if st.button(suggestions[2], use_container_width=True): st.session_state.selected_question = suggestions[2]
    if st.button(suggestions[3], use_container_width=True): st.session_state.selected_question = suggestions[3]

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask about my experience...")
if st.session_state.selected_question:
    user_input = st.session_state.selected_question
    st.session_state.selected_question = None

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Searching experience and generating answer..."):
            try:
                docs = similarity_search(user_input, store, embedder)
                context = "\n\n---\n\n".join(
                    f"[Source: {d['source']}]\n{d['text']}" for d in docs
                )
                response = chain.invoke({"context": context, "question": user_input})
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")

st.divider()
st.caption("Built with LangChain + Numpy + Groq | [View source](https://github.com/krisdam) | RAG-powered career Q&A")
