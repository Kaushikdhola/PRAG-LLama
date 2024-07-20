from langchain_community.llms import Ollama
from app.faiss_db import FaissDB
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def initialize_models():
    llama_model = Ollama(model="phi3", temperature = 0.1)
    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_texts(["Initial document"], embeddings)
    faiss_db = FaissDB() 
    
    return llama_model, vector_store, faiss_db, embeddings
