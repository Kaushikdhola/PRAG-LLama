from langchain_community.llms import Ollama
from transformers import AutoTokenizer, AutoModel
from app.faiss_db import FaissDB
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import util
import streamlit as st

def generate_prompt(user_input, user_details, last_response, llama_model, vector_store, embeddings):


    # Create a prompt template
    prompt_template = PromptTemplate(
    input_variables=["user_details", "user_input", "context"],
    template="""
    User details: {user_details}

    Question: {user_input}

    Context: {context}
    
    Please create a specific and effective prompt based on the user's details, the user's question, and the provided context. If the context is relevant to the question, include it in the prompt. Otherwise, generate the prompt using just the user's details and question. Here is the prompt I need:
    """
    )


    # Initialize context
    context = ""
    sim_score = 0
    user_input_embedding = embeddings.embed_query(user_input)

    # Check similarity with last response
    if last_response:
        last_response_embedding = embeddings.embed_query(last_response)
        sim_score = util.cos_sim(last_response_embedding, user_input_embedding).item()
        print("Sim Score for last response {}".format(sim_score))
        # Only add the last response to context if similarity is above 0.8
        if sim_score > 0.9:
            context += f"Previous response: {last_response}\n"

    # Perform RAG (Retrieval-Augmented Generation)
    relevant_docs = vector_store.similarity_search(user_input, k=1)
    print(relevant_docs)
    
    # Add relevant information only if similarity condition is met
    for doc in relevant_docs:

        doc_embedding = embeddings.embed_query(doc.page_content)
        sim_score_doc = util.cos_sim(doc_embedding, user_input_embedding).item()
        print("Sim Score for doc response {} : {}".format(doc.page_content,sim_score_doc))
        
        if sim_score_doc > 0.13:
            context += f"Our Previous Conversation which is related to the asked question: {doc.page_content}\n"

    # Create an LLMChain
    llm_chain = LLMChain(
        llm=llama_model,
        prompt=prompt_template,
        verbose=True
    )
    
    details = user_details
    if st.session_state.chat_history and len(context)!=0:
        details = None
    
    # Generate response
    response = llm_chain.predict(user_details=details, user_input=user_input, context=context)
    
    print("-------------------Response----------------", response)

    # # Generate a vector for the response and store it
    # response_vector = embeddings.embed_query(response)
    # vector_store.add_texts([response], embeddings=[response_vector])

    return response