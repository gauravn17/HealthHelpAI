# 🏥 Medical Billing Assistant

A Streamlit-powered AI assistant designed to help medical billers and healthcare professionals quickly understand and answer questions about medical billing documents, codes, and procedures.

## 🚀 Features

- 📄 Upload and parse medical billing PDFs
- 🤖 Ask natural language questions about your document
- 💬 Powered by LangChain and LLMs for contextual QA
- 🧠 Custom-trained on medical billing knowledge
- 🌐 Web-based interface via Streamlit

## 📦 Tech Stack

- **Frontend**: Streamlit
- **Backend**: LangChain, Python
- **LLM**: OpenAI / Local model
- **Document Parsing**: PyPDF
- **Vector Store**: FAISS / ChromaDB

- **Model**: google/flan-t5-base
- **Provider**: Hugging Face (running locally via transformers.pipeline)
- **Type**:Text-to-text generation model (fine-tuned for instruction following)

🧠 AI Model Role in this app :
	•	The FLAN-T5 model generates answers based on the top 4 most relevant text chunks retrieved from the vector database (Chroma with sentence-transformers/all-MiniLM-L6-v2 embeddings).
	•	It’s ideal for question answering, summarization, and instruction-based NLP tasks.
