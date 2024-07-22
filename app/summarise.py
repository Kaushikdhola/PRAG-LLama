from langchain_community.llms import Ollama
from transformers import AutoTokenizer, AutoModel
from app.faiss_db import FaissDB
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import util

def summarise(user_question, llama_response, phi3_response, model):
    
    prompt_template = PromptTemplate(
    input_variables=["user_question", "llama_response", "phi3_response"],
    template="""
    Based on my details, please summarize the following responses to the question: "{user_question}"

    LLama model response: {llama_response}

    Phi3 model response: {phi3_response}

    Please provide a concise summary that combines the key points from both responses.
    """
    )
    
    llm_chain = LLMChain(
        llm=model,
        prompt=prompt_template,
        verbose=True
    )

    # Generate response
    response = llm_chain.predict(user_question = user_question, llama_response = llama_response, phi3_response = phi3_response)
    
    print("-------------------Summarise Response----------------", response)
    
    return response
    
    
