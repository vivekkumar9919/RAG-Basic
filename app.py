import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

def query_rag(question):
    persist_dir = "chroma_db"
    
    # 1. Check if DB exists
    if not os.path.exists(persist_dir):
        print(f"Error: Vector database '{persist_dir}' not found. Please run 'python ingest.py' first.")
        return

    print("1. Initializing local embeddings and LLM (llama3.2:1b)...")
    embeddings = OllamaEmbeddings(model="llama3.2:1b")
    llm = Ollama(model="llama3.2:1b", temperature=0.0)

    print("2. Loading Chroma DB and preparing Hybrid Search...")
    vectorstore = Chroma(
        persist_directory=persist_dir, 
        embedding_function=embeddings
    )
    
    # --- PRO TIP: Hybrid Search ---
    # Small 1B models often have "fuzzy" semantic memory. 
    # Implementing a Keyword fallback (Hybrid Search) ensures that specific 
    def hybrid_retriever_func(query):
        # A. Semantic Search (The standard vector search)
        semantic_docs = vectorstore.similarity_search(query, k=5)
        semantic_texts = {d.page_content for d in semantic_docs}
        
        # B. Keyword Search (A direct scan for exact words in the query)
        all_res = vectorstore.get()
        all_texts = all_res["documents"]
        all_metas = all_res["metadatas"]
        
        keywords = query.lower().split()
        keyword_docs = []
        for text, meta in zip(all_texts, all_metas):
            if any(k in text.lower() for k in keywords if len(k) > 3):
                keyword_docs.append(Document(page_content=text, metadata=meta))
        
        keyword_texts = {d.page_content for d in keyword_docs[:5]}
        
        # Combine and identify sources
        combined = keyword_docs[:5] + semantic_docs
        seen = set()
        unique_docs = []
        
        print("\n--- RETRIEVAL ANALYSIS ---")
        for d in combined:
            if d.page_content not in seen:
                source = "???"
                in_semantic = d.page_content in semantic_texts
                in_keyword = d.page_content in keyword_texts
                
                if in_semantic and in_keyword:
                    source = "BOTH (Semantic + Keyword)"
                elif in_semantic:
                    source = "CHROMA (Semantic Search)"
                elif in_keyword:
                    source = "MANUAL (Keyword Scan)"
                
                # print(f"Found Chunk: {d.page_content[:100]}... [Source: {source}]")
                print(f"Found Chunk: [Source: {source}]")
                unique_docs.append(d)
                seen.add(d.page_content)
        
        print(f"Total unique chunks found: {len(unique_docs)}")
        return unique_docs

    print("3. Setting up RAG Chain...")
    system_prompt = (
        "### INSTRUCTION:\n"
        "You are a helpful AI assistant. Use the provided Context below to answer the user's question.\n"
        "1. Stay strictly within the context provided.\n"
        "2. If the context contains information about the topic (e.g., headings or brief mentions), summarize what is written even if it is brief.\n"
        "3. If the answer is completely missing, say: 'I'm sorry, but that information is not available in my current documents.'\n"
        "\n### CONTEXT:\n{context}\n"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Wrap functions for LCEL pipe support
    retriever_runnable = RunnableLambda(hybrid_retriever_func)
    format_runnable = RunnableLambda(format_docs)

    rag_chain = (
        {"context": retriever_runnable | format_runnable, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n" + "="*40)
    print(f"QUERY: {question}")
    print("="*40 + "\n")
    
    response = rag_chain.invoke(question)

    print("--- Local LLM Answer ---")
    print(response)

if __name__ == "__main__":
    user_question = "Explain Production-Ready RAG"
    query_rag(user_question)