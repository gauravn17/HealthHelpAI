import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from datetime import datetime

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline

load_dotenv()

# --- Load the PDF, Split, and Embed ---
@st.cache_resource
def setup_rag_pipeline(pdf_path, persist_directory="vectorstore"):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(chunks, embedding, persist_directory=persist_directory)

    # Local LLM pipeline (Flan-T5)
    local_llm = pipeline("text2text-generation", model="google/flan-t5-base", tokenizer="google/flan-t5-base", max_length=512)
    llm = HuggingFacePipeline(pipeline=local_llm)

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever(search_type="similarity", k=4))
    return qa_chain

# --- Initialize Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "start_time" not in st.session_state:
    st.session_state.start_time = datetime.now()

# --- Page Config ---
st.set_page_config(
    page_title="Medical Billing Assistant",
    page_icon="🩺",
    layout="wide"
)

# --- Sidebar ---
with st.sidebar:
    st.title("🩺 Medical Billing Assistant")
    st.markdown("---")
    
    # Statistics
    st.subheader("📊 Statistics")
    st.metric("Total Queries", st.session_state.query_count)
    session_duration = datetime.now() - st.session_state.start_time
    st.metric("Session Duration", f"{session_duration.seconds // 60} minutes")
    
    # Settings
    st.markdown("---")
    st.subheader("⚙️ Settings")
    model_name = st.selectbox(
        "Model",
        ["google/flan-t5-base"],
        index=0
    )
    
    # About
    st.markdown("---")
    st.subheader("ℹ️ About")
    st.markdown("""
    This assistant helps you understand your medical benefits plan.
    Ask questions about coverage, costs, and more.
    """)

# --- Main Content ---
col1, col2 = st.columns([2, 1])

with col1:
    st.title("Chat Interface")
    
    # Chat Form
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area("Ask a question:", height=100)
        submit = st.form_submit_button("Submit", use_container_width=True)
    
    if submit and user_input:
        with st.spinner("Thinking..."):
            response = qa_chain.invoke(user_input)
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Assistant", response["result"]))
            st.session_state.query_count += 1
    
    # Chat History
    st.markdown("---")
    for speaker, msg in st.session_state.chat_history:
        if speaker == "You":
            st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin: 5px 0;'>
                <strong>🧑 You:</strong> {msg}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background-color: #e6f3ff; padding: 10px; border-radius: 10px; margin: 5px 0;'>
                <strong>🤖 Assistant:</strong> {msg}
            </div>
            """, unsafe_allow_html=True)

with col2:
    st.title("Insights")
    
    # Recent Activity
    st.subheader("📈 Recent Activity")
    if st.session_state.chat_history:
        recent_queries = pd.DataFrame([
            {"Query": msg[1], "Type": msg[0]} 
            for msg in st.session_state.chat_history[-6:]
        ])
        st.dataframe(recent_queries, use_container_width=True)
    
    # Quick Tips
    st.subheader("💡 Quick Tips")
    st.markdown("""
    - Ask about specific coverage details
    - Inquire about costs and deductibles
    - Get information about network providers
    - Understand your benefits and limitations
    """)
    
    # Document Status
    st.subheader("📄 Document Status")
    pdf_path = "docs/sbc-completed-sample.pdf"
    if os.path.exists(pdf_path):
        st.success("✅ Benefits document loaded successfully")
    else:
        st.error("❌ PDF not found. Place it in the 'docs' folder.")
        st.stop()

# Initialize QA Chain
qa_chain = setup_rag_pipeline(pdf_path)