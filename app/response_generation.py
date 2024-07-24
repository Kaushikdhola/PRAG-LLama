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

# def initialize_models():
#     llama_model = Ollama(model="llama3")
#     embeddings = HuggingFaceEmbeddings()
#     vector_store = FAISS.from_texts(["Initial document"], embeddings)
#     faiss_db = FaissDB() 
    
#     return llama_model, vector_store, faiss_db, embeddings


# def generate_response(user_input, user_details, last_response, llama_model, vector_store):
def generate_response(user_input, context, phi_model):
    embeddings = HuggingFaceEmbeddings()

    # Create a prompt template
    # prompt_template = PromptTemplate(
    #     input_variables=["user_details", "user_input", "context"],
    #     template="""
    #     My details: {user_details}

    #     Question: {user_input}

    #     Context: {context}
        
    #     Please create a specific and effective prompt based on my details, my question, and the provided context. If the context is relevant to the question, include it in the prompt. Otherwise, generate the prompt using just my details and question. Here is the prompt I need:
    #     """
    # )
    
    prompt_template = PromptTemplate(
    input_variables=["generated_prompt"],
    template="""

    Based on my details, please answer the following question:

    {generated_prompt}
    """
    )


    # Initialize context
    # context = ""
    # sim_score = 0
    # user_input_embedding = embeddings.embed_query(user_input)

    # # Check similarity with last response
    # if last_response:
    #     last_response_embedding = embeddings.embed_query(last_response)
    #     sim_score = util.cos_sim(last_response_embedding, user_input_embedding).item()
    #     print("Sim Score for last response {}".format(sim_score))
    #     # Only add the last response to context if similarity is above 0.8
    #     if sim_score > 0.9:
    #         context += f"Previous response: {last_response}\n"

    # # Perform RAG (Retrieval-Augmented Generation)
    # relevant_docs = vector_store.similarity_search(user_input, k=1)
    # print(relevant_docs)
    
    # # Add relevant information only if similarity condition is met
    # for doc in relevant_docs:
    #     doc_embedding = embeddings.embed_query(doc.page_content)
    #     sim_score_doc = util.cos_sim(doc_embedding, user_input_embedding).item()
    #     print("Sim Score for doc response {} : {}".format(doc.page_content,sim_score_doc))
    #     context += f"Our Previous Conversation which is related to the asked question: {doc.page_content}\n"

    # Create an LLMChain
    llm_chain = LLMChain(
        llm=llama_model,
        prompt=prompt_template,
        verbose=True
    )

    # Generate response
    response = llm_chain.predict(generated_prompt = user_input)
    
    print("-------------------Response----------------", response)

    # Generate a vector for the response and store it

    return response
