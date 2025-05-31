# utils.py

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

import qdrant_client
import os

# Load and split PDF
def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    return pages

def chunk_text(pages, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(pages)
    return chunks

# Embed and store in Qdrant
def embed_and_store(chunks, collection_name="pdf_chunks"):
    # Initialize embeddings (CPU-friendly model)
    embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Qdrant client (assuming local server at localhost:6333)
    client = qdrant_client.QdrantClient("http://localhost:6333")

    # Store documents in Qdrant
    vectorstore = Qdrant.from_documents(
        documents=chunks,
        embedding=embed_model,
        url="http://localhost:6333",
        collection_name=collection_name
    )
    return vectorstore

# Retrieve top-k relevant chunks
def retrieve_context(query, vectorstore, k=3):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever

# Ask the LLM
def ask_llm(query, retriever):
    llm = Ollama(model="llama3.2")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    result = qa_chain.invoke({"query": query})
    return result["result"]

from gtts import gTTS
import base64
import os

def speak_text(text, lang="en"):
    tts = gTTS(text, lang=lang)
    audio_path = "response.mp3"
    tts.save(audio_path)
    
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    # Delete the temp file after reading
    os.remove(audio_path)

    # Return base64 audio data
    b64 = base64.b64encode(audio_bytes).decode()
    return f"data:audio/mp3;base64,{b64}"
