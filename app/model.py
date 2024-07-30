from langchain_community.llms import Ollama
from app.VectoreStore import VectorStore


def initialize_models():
    llama_model = Ollama(model="llama3", temperature = 0, format = "json")
    phi_model = Ollama(model="phi3", temperature = 0, format="json")    
    vector_store = VectorStore()

    return llama_model, vector_store, phi_model
