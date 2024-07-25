from langchain_community.llms import Ollama
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def initialize_models():
    llama_model = Ollama(model="phi3", temperature = 0.1)
    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_texts(["Initial document", "I love coding in Java"], embeddings)
    return llama_model, vector_store, embeddings