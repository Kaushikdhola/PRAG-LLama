import streamlit as st
from .user_details import collect_user_details
from .chat import display_chat, handle_user_input
from .response_generation import generate_response
from .model import initialize_models
from .prompt_generation import generate_prompt
from .trainRAG.main import train_model_page

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
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        if 'user_details' not in st.session_state:
            st.session_state.user_details = None

        if 'models' not in st.session_state:
            st.session_state.models = initialize_models()

        # Collect user details
        collect_user_details()

        # Unpack models
        llama_model, vector_store, faiss_db, embeddings = st.session_state.models

        # Handle user input
        user_input = handle_user_input()

        # Process user input and generate response
        if user_input:
            last_response = st.session_state.chat_history[-1]['bot'] if st.session_state.chat_history else None
            user_details = st.session_state.user_details
            prompt = generate_prompt(user_input, user_details, last_response, llama_model, vector_store, embeddings)
            response = generate_response(prompt, llama_model)
            
            st.session_state.chat_history.append({"user": user_input, "bot": response})
            response_vector = embeddings.embed_query(response)
            vector_store.add_texts([response], embeddings=[response_vector])

        display_chat()

       