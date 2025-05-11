# 🏥 HealthHelp AI: Medical Billing Assistant

**HealthHelp AI** is an intelligent assistant designed to simplify the complexities of medical billing. Built with Streamlit, LangChain, and Hugging Face models, it enables users to upload billing PDFs and ask natural language questions — receiving accurate, context-aware responses in seconds.

---

## 🚀 Features

- 📄 **Document Upload** – Upload and process standard medical billing PDFs
- 🧠 **AI-Powered Understanding** – Converts documents into searchable knowledge
- 🤖 **Natural Language QA** – Ask plain English questions and get answers instantly
- 💬 **Contextual Awareness** – Retrieves relevant sections from documents for better answers
- 🔍 **Semantic Search** – Uses embeddings to find similar content
- 🌐 **Web Interface** – Fully interactive, easy-to-use Streamlit front end
- 📌 **Offline-Compatible** – Runs locally with no need for cloud APIs (privacy-safe)

---

## 📦 Tech Stack

| Layer         | Technology                                 |
|--------------|---------------------------------------------|
| 🖥️ Frontend  | [Streamlit](https://streamlit.io/)          |
| 🧠 Backend    | [LangChain](https://www.langchain.com/), Python |
| 📄 Parsing    | [PyPDFLoader](https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf) |
| 📚 Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| 🗃️ Vector DB | [Chroma](https://www.trychroma.com/)        |
| 🤖 Model      | `google/flan-t5-base` (via Hugging Face Transformers) |

---

## 🧠 AI Model Details

- **Model**: `google/flan-t5-base`
- **Provider**: Hugging Face (local)
- **Type**: Text-to-text generation, instruction-tuned
- **Use Case Fit**: QA over long, structured documents (ideal for CPT codes, invoice summaries, insurance notes)

### 🔍 Model Role in HealthHelp AI:
- Retrieves the **top 4 relevant chunks** of your uploaded document using semantic similarity (Chroma vector store + MiniLM embeddings)
- Feeds those chunks into `flan-t5-base` to generate a **concise and accurate answer**
- Enables powerful tasks like:
  - Explaining billing codes
  - Summarizing insurance charges
  - Identifying CPT/ICD-10 codes
  - Flagging inconsistencies

---

## 🛠️ How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/gauravn17/medical-billing-assistant.git
   cd medical-billing-assistant
