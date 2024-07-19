import streamlit as st
from .user_details import collect_user_details
from .chat import display_chat, handle_user_input
from .response_generation import initialize_models, generate_response

def run_app():
    st.title("Personalized Recommender System")

    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'user_details' not in st.session_state:
        st.session_state.user_details = None

    # Collect user details
    collect_user_details()

    # Initialize models
    llama_model, faiss_db = initialize_models()

    # Handle user input
    user_input = handle_user_input()

    # Process user input and generate response
    if user_input:
        last_response = st.session_state.chat_history[-1]['bot'] if st.session_state.chat_history else None
        user_details = st.session_state.user_details
        response = generate_response(user_input, user_details, last_response, llama_model, faiss_db)
        
        st.session_state.chat_history.append({"user": user_input, "bot": response})

    display_chat()
