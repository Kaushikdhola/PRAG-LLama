import streamlit as st
from .user_details import collect_user_details
from .chat import display_chat, handle_user_input
from .response_generation import generate_response
from .model import initialize_models
from .prompt_generation import generate_prompt

def train_model_page():
    st.title("Train Your Own Model")
    
    if 'training_input' not in st.session_state:
        st.session_state.training_input = ""

    # Text input for training data
    user_input = st.text_area("Enter your training data (press Enter to add more content):", 
                              value=st.session_state.training_input,
                              height=200,
                              key="training_data_input")
    
    # Button to append content to file
    if st.button("Add Content"):
        with open("training_data.txt", "a") as f:
            f.write(user_input + "\n")
        st.success("Content added successfully!")
    
    # Button to train the model
    if st.button("Train Model"):
        # Read the entire file
        with open("training_data.txt", "r") as f:
            training_data = f.read()
        
        # Here you would implement training logic - kunal
        # For this example, we'll just pretend to train
        st.info("Training model... This may take a while.")
        # Simulating training time
        import time
        time.sleep(5)
        st.success("Model trained successfully!")
    
    # Input for questions
    question = st.text_input("Ask a question based on the trained data:")
    
    # Button to submit question
    if st.button("Submit Question"):
        if 'models' not in st.session_state:
            st.session_state.models = initialize_models()
        
        llama_model, vector_store, faiss_db, embeddings = st.session_state.models
        
        # Here you would implement personalized RAG logic - kunal
        # For this example, we'll just use a simple response
        prompt = generate_prompt(question, {}, None, llama_model, vector_store, embeddings)
        response = generate_response(prompt, llama_model)
        
        st.write("Response:", response)

    if st.button("Return to Main Page"):
        st.session_state.page = "main"
        st.experimental_rerun()

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

       