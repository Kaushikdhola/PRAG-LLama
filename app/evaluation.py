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


prompt_template = PromptTemplate(
    input_variables=["question", "answer"],
    template="""
    You will be given a user_question and system_answer couple.
    Your task is to provide a 'total rating' scoring how well the system_answer answers the user concerns expressed in the user_question.
    Give your answer on a scale of 1 to 4, where 1 means that the system_answer is not helpful at all, and 4 means that the system_answer completely and helpfully addresses the user_question.
    Here is the scale you should use to build your answer:
    1: The system_answer is terrible: completely irrelevant to the question asked, or very partial
    2: The system_answer is mostly not helpful: misses some key aspects of the question
    3: The system_answer is mostly helpful: provides support, but still could be improved
    4: The system_answer is excellent: relevant, direct, detailed, and addresses all the concerns raised in the question
    Provide your feedback as follows:
    Feedback:::
    Evaluation: (your rationale for the rating, as a text)
    Total rating: (your rating, as a number between 1 and 4)
    You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.
    Now here are the question and answer.
    Question: {question}
    Answer: {answer}
    Provide your feedback.
    """
)

def evaluate_response(user_input, response, vector_store, model):
    
    # model = Ollama(model="llama3", temperature = 0.0)

    judge_model = LLMChain(
        llm=model,
        prompt=prompt_template,
        verbose=True
    )
    
    eval_result = judge_model.predict(question = user_input, answer = response)
    
    print("-------------------Evaluation Response-------------------", eval_result)
    
    return eval_result

    
