---
name: RAG Pipeline Specialist
description: Expert in local RAG architectures using Ollama, ChromaDB, and LangChain.
---

# RAG Skills for Local Development

This skill provides patterns and instructions for maintaining and improving the local RAG pipeline in this repository.

## 🏗️ Architecture
- **Environment**: Local Ollama (llama3.2:1b)
- **Vector Store**: ChromaDB
- **Ingestion**: `ingest.py` (Custom chunking for technical PDF parsing)
- **Retrieval**: `app.py` (Multi-Query Retrieval)

## 🛠️ Key Techniques

### 1. Multi-Query Retrieval
When using small models like `llama3.2:1b`, semantic search can often miss specific sections. We use `MultiQueryRetriever` to generate multiple perspectives of a question, which increases the "search surface area" and improves recall for technical headings.

### 2. Precise Ingestion
For technical surveys and papers:
- **Chunk Size**: 500 characters (Smaller chunks preserve context for headings)
- **Overlap**: 100 characters (Reduces the risk of splitting vital sentences)

### 3. Hallucination Prevention
The system prompt is designed to be helpful but strict. It requires the model to:
- Acknowledge when information is missing.
- Summarize brief mentions rather than ignoring them.
- Stick exclusively to the provided context.

## 🚀 Troubleshooting
- **No information found**: If a clear heading is missed, consider increasing `k` in the retriever or checking the PDF extraction logs.
- **Generic answers**: Ensure the temperature is set to `0.0` for deterministic and factual responses.
- **Speed**: Ingestion of ~15 pages takes about 2-5 minutes locally on a 1B model.
