from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.faiss_db import FaissDB
import numpy as np


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def initialize_models():
    llama_model = Ollama(model="llama3") 
    faiss_db = FaissDB() 
    return llama_model, faiss_db

def calculate_similarity(text1, text2):
    embeddings1 = embedding_model.embed_query(text1)
    embeddings2 = embedding_model.embed_query(text2)
    return np.dot(embeddings1, embeddings2) / (np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2))

def generate_response(user_input, user_details, last_response, llama_model, faiss_db):
    # Construct the prompt with user details
    prompt = f"User details: {user_details}\n\nUser input: {user_input}"
    
    if last_response:
        sim_score = calculate_similarity(user_input, last_response)
        if sim_score > 0.8:
            prompt += f" Previous response: {last_response}"
    
    # Generate a response using the llama model
    response = llama_model.generate(prompt=prompt)
    
    # Generate a vector for the response
    response_vector = embedding_model.embed_query(response)
    
    # Store the vector in FaissDB
    faiss_db.store_vector(response_vector)
    
    return response