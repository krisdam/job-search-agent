"""
Streamlit Frontend — Career Q&A
"""
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

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
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
    return FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)

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

def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[Source: {os.path.basename(d.metadata.get('source','unknown'))}]\n{d.page_content}"
        for d in docs
    )

@st.cache_resource
def build_chain():
    vectorstore = load_vectorstore()
    llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT | llm | StrOutputParser()
    )
    return chain, vectorstore

chain, vectorstore = build_chain()
st.caption(f"📚 Knowledge base: {vectorstore.index.ntotal} embedded chunks")

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
                response = chain.invoke(user_input)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")

st.divider()
st.caption("Built with LangChain + FAISS + Groq | [View source](https://github.com/krisdam) | RAG-powered career Q&A")
