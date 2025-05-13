import os
import sys
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

# === Utility functions ===
def bold(text):
    return f"\033[1m{text}\033[0m"

def load_and_split_docs(file_path):
    print("📄 Loading and splitting document...")
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Parsed document returned no chunks.")
    return chunks

def create_or_load_vectorstore(chunks, persist_directory):
    print("🧠 Creating or loading vectorstore with HuggingFace embeddings...")
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    else:
        vectordb = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory=persist_directory)

    return vectordb

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

# === Entry Point ===
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❗ Usage: python backend.py <path_to_pdf>")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at '{file_path}'")
        sys.exit(1)

    try:
        chunks = load_and_split_docs(file_path)

        # Use filename to persist separate vectorstore
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        persist_directory = f"vectorstore/{base_name}"

        vectordb = create_or_load_vectorstore(chunks, persist_directory)
        qa_chain = build_qa_chain(vectordb)

        print(bold("\n🩺 HealthHelp AI Ready! Ask your questions below."))
        print("(Type 'exit' to quit)\n")

        while True:
            question = input("❓ You: ")
            if question.lower().strip() in ["exit", "quit"]:
                print("👋 Exiting assistant. Take care!")
                break

            response = qa_chain.invoke(question)
            print(f"💡 Answer: {response['result']}\n")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
