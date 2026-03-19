# Basic RAG with Local LLM (Ollama)

A high-accuracy Retrieval-Augmented Generation (RAG) pipeline designed for local development using Ollama and ChromaDB.

## 🚀 Features
- **Local LLM**: Powered by `llama3.2:1b` (via Ollama).
- **Hybrid Retrieval**: Combines **Semantic Search** (Vector similarity) and **Keyword Search** (Manual scan) to ensure technical headings like "Scaling Laws" are never missed.
- **Split Pipeline**: Clean separation between document ingestion (`ingest.py`) and querying (`app.py`).
- **Strict Anti-Hallucination**: A custom system prompt that ensures the model sticks strictly to the provided PDF context.

## 🛠️ Prerequisites
- [Ollama](https://ollama.com/) installed and running.
- Pull the model: `ollama pull llama3.2:1b`.
- Python 3.10+ installed.

## 📦 Setup

1. **Clone the repository**:
   ```bash
   git clone git@github.com:vivekkumar9919/RAG-Basic.git
   cd RAG-Basic
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install langchain langchain-chroma langchain-community pypdf python-dotenv
   ```

4. **Environment Variables**:
   Create a `.env` file in the root directory if you need specific configuration (though not strictly required for local Ollama).

## 🏃 How to Run

### Step 1: Ingest Documents
Place your PDF in the `data/` folder (default is `data/rag.pdf`) and run the ingestion script. This creates a local vector database in the `chroma_db/` folder.
```bash
python ingest.py
```

### Step 2: Query the RAG
Run the query script to ask questions about your document.
```bash
python app.py
```

## 🧠 Advanced Retrieval
This project uses a **Hybrid Search** pattern. If the semantic embedding model fails to rank a specific heading highly, the system automatically performs a keyword scan to ensure the most relevant chunks are always included in the LLM's context.
