from langchain_community.llms import Ollama
from transformers import AutoTokenizer, AutoModel
from app.faiss_db import FaissDB
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import util

def initialize_models():
    llama_model = Ollama(model="llama3")
    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_texts(["Initial document"], embeddings)
    faiss_db = FaissDB() 
    
    return llama_model, vector_store, faiss_db


def generate_response(user_input, user_details, last_response, llama_model, vector_store):
    embeddings = HuggingFaceEmbeddings()

    # Create a prompt template
    prompt_template = PromptTemplate(
        input_variables=["user_details", "user_input", "context"],
        template="""
        User details: {user_details}

        User input: {user_input}

        Context: {context}

        Based on the user details, their input, and the provided context, please generate a personalized response:
        """
    )

    # Initialize context
    context = ""
    sim_score = 0
    # Check similarity with last response
    if last_response:
        last_response_embedding = embeddings.embed_query(last_response)
        user_input_embedding = embeddings.embed_query(user_input)
        sim_score = util.cos_sim(last_response_embedding, user_input_embedding).item()
        
        # Only add the last response to context if similarity is above 0.8
        if sim_score > 0.9:
            context += f"Previous response: {last_response}\n"

    # Perform RAG (Retrieval-Augmented Generation)
    relevant_docs = vector_store.similarity_search(user_input, k=3)
    
    # Add relevant information only if similarity condition is met
    if sim_score > 0.8:
        for doc in relevant_docs:
            context += f"Relevant information: {doc.page_content}\n"

    # Create an LLMChain
    llm_chain = LLMChain(
        llm=llama_model,
        prompt=prompt_template,
        verbose=True
    )

    # Generate response
    response = llm_chain.predict(user_details=user_details, user_input=user_input, context=context)

    # Generate a vector for the response and store it
    response_vector = embeddings.embed_query(response)
    vector_store.add_texts([response], embeddings=[response_vector])

    return response
