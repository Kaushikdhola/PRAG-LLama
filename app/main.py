import streamlit as st
import concurrent.futures
from .user_details import collect_user_details
from .chat import display_chat, handle_user_input
from .response_generation import generate_response
from .model import initialize_models
from .prompt_generation import generate_prompt
from .trainRAG.main import train_model_page
from .evaluation import evaluate_response   
from .summarise import summarise
# from langchain.retrievers.multi_query import MultiQueryRetriever
from .multiqueryretreiver import multiQueryRetreiver
from .perAnalyse import personalize_analyzer

def run_app():
    if st.button("Train Your Own Model"):
        st.session_state.page = "train"

    if 'page' not in st.session_state:
        st.session_state.page = "main"
    
    if st.session_state.page == "train":
        train_model_page()
    else:
        st.title("Personalized Recommender System")

        # Initialize session state
        # if 'chat_history' not in st.session_state:
        #     st.session_state.chat_history = []

        if 'user_details' not in st.session_state:
            st.session_state.user_details = None

        if 'models' not in st.session_state:
            st.session_state.models = initialize_models()

        # Collect user details
        collect_user_details()

        # Unpack models
        llama_model, vector_store, faiss_db, embeddings, phi_model = st.session_state.models

        # Handle user input
        user_input = handle_user_input()

        # Process user input and generate response
        if user_input:
            # last_response = st.session_state.chat_history[-1]['bot'] if st.session_state.chat_history else None
            user_details = st.session_state.user_details
                
            isPersonalised=personalize_analyzer(user_input, phi_model, vector_store, embeddings)
            if (isPersonalised=="yes"):
                print("yes personlized")
                vector_store.add_texts([user_input], embeddings=[embeddings.embed_query(user_input)])
                
            for i in range(3):
                context = multiQueryRetreiver(user_input, phi_model, vector_store)
                
                response = generate_response(user_input, context, phi_model)
                
                evaluation_score = evaluate_response(user_input, response, vector_store, llama_model)
            
                if evaluation_score > threshold: 
                    break
            
            
            # prompt = generate_prompt(user_input, user_details, lazst_response, llama_model, vector_store, embeddings)
            
            # Define a helper function for generating responses
            # def generate_responses():
            #     with concurrent.futures.ThreadPoolExecutor() as executor:
            #         future_response_1 = executor.submit(generate_response, prompt, llama_model)
            #         future_response_2 = executor.submit(generate_response, prompt, phi_model)
            #         response_1 = future_response_1.result()
            #         response_2 = future_response_2.result()
            #     return response_1, response_2

            # # Get responses in parallel
            # response_1, response_2 = generate_responses()
            
            # st.write(response_1, response_2)
            
            # Summarize the responses
            # summarize_response = summarise(user_input, response_1, response_2, llama_model)
            # eval = evaluate_response(user_input, summarize_response, llama_model)
            
            # Update chat history and vector store
            # st.session_state.chat_history.append({"user": user_input, "bot": summarize_response})
            # response_vector = embeddings.embed_query(summarize_response)
            # vector_store.add_texts([summarize_response], embeddings=[response_vector])
            
            # # Display results
            # st.write("Summarise Result:")
            # st.write(summarize_response)
            
            # st.write("Evaluation Results:")
            # st.write(eval)
            
        display_chat()  
