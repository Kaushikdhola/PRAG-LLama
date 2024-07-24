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
from langchain.retrievers.multi_query import MultiQueryRetriever
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)



def multiQueryRetreiver(user_query, phi_model, vectorStore):
    
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever = vectorStore.as_retriever(), llm = phi_model
    )
    
    docs = retriever_from_llm.invoke(user_query)
    
    context = "\n---\n".join([d.page_content for d in docs])
    
    
    print("-" * 100)
    
    print(context)
    
    return context
    

    
    