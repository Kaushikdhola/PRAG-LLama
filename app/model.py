from langchain_community.llms import Ollama
from app.faiss_db import FaissDB
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def initialize_models():
    llama_model = Ollama(model="llama3", temperature = 0)
    phi_model = Ollama(model="phi3", temperature = 0)
    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_texts(["Initial document", "My name is kunal makwana", "I love coding in java", "I am vegetarian", "I like machine learning", "I like to play games", "I am good at counter strike", "I am 23 years old", "I am studying at Dalhousie University", "I am living in halifax"], embeddings)
    faiss_db = FaissDB() 
    
    return llama_model, vector_store, faiss_db, embeddings, phi_model
