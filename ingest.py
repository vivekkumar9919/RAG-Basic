import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Load environment variables
load_dotenv()

def ingest_docs():
    persist_dir = "chroma_db"
    
    # 1. Clear old DB for a fresh start with new chunking!
    import shutil
    if os.path.exists(persist_dir):
        print(f"Clearing old {persist_dir} to apply new chunking settings...")
        shutil.rmtree(persist_dir)

    print("1. Loading document...")
    loader = PyPDFLoader("data/rag.pdf")
    docs = loader.load()

    print("2. Splitting document into SMALLER chunks (500 chars) for better precision...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splitted_docs = text_splitter.split_documents(docs)

    print(f"3. Creating embeddings and storing in Chroma DB ({len(splitted_docs)} chunks)...")
    embeddings = OllamaEmbeddings(model="llama3.2:1b")
    
    vectorstore = Chroma.from_documents(
        documents=splitted_docs, 
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print(f"4. Successfully created Chroma DB in '{persist_dir}'. length of documents is {len(splitted_docs)}")

if __name__ == "__main__":
    ingest_docs()
