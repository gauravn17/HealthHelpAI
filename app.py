import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

# Load environment variables
load_dotenv()

# Function to load and split PDF into chunks
def load_and_split_docs(file_path):
    print("📄 Loading and splitting document...")
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    return chunks

# Function to create or load a vector store with embeddings
def create_or_load_vectorstore(chunks, persist_directory="vectorstore"):
    print("🧠 Creating or loading vectorstore with HuggingFace embeddings...")
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    else:
        vectordb = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory=persist_directory)

    return vectordb

# Function to build the QA chain using a local HuggingFace model
def build_qa_chain(vectordb):
    retriever = vectordb.as_retriever(search_type="similarity", k=4)

    local_pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        tokenizer="google/flan-t5-base",
        max_length=512,
        temperature=0.5
    )

    llm = HuggingFacePipeline(pipeline=local_pipe)

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

# === TOP LEVEL EXECUTION FOR STREAMLIT COMPATIBILITY ===

file_path = "docs/sbc-completed-sample.pdf"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ Error: File not found at {file_path}. Place your PDF in the 'docs' folder.")

chunks = load_and_split_docs(file_path)
vectordb = create_or_load_vectorstore(chunks)
qa_chain = build_qa_chain(vectordb)

# === MAIN INTERFACE LOOP FOR CLI USAGE ===

if __name__ == "__main__":
    print("\n🩺 Medical Billing Assistant Ready! Ask your questions below.\n(Type 'exit' to quit)\n")

    while True:
        question = input("❓ You: ")
        if question.lower() in ["exit", "quit"]:
            print("👋 Exiting assistant. Take care!")
            break

        answer = qa_chain.invoke(question)
        print("💡 Answer:", answer, "\n")