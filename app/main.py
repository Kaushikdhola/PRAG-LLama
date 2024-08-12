import streamlit as st
from .user_details import collect_user_details
from .chat import display_chat, handle_user_input
from .response_generation import generate_response
from .model import initialize_models
from .evaluation import evaluate_response   
from .multiqueryretreiver import multiQueryRetreiver
from .perAnalyse import personalize_analyzer
from .docRetreiver import docRetreiver

EVAL_THRESHOLD = 0.7
RETRY = 3

def try_generate(user_input, context, phi_model, llama_model, feedback = ""):
    response = generate_response(user_input, context, phi_model, feedback)
    print("Response : ", response)
    
    if response != user_input:
        Q, K, V = user_input, context, response
                    
        score, reasoning = evaluate_response(Q, K, V, llama_model)
        
        return V, score, reasoning
    
    # returning 1 as response and user_input is same
    return user_input, 1, ""
        
def formatFeedback(response, feedback):
    return """
    Evaluation of rephrased question
    
    Rephrased Question: {response}
    
    Feedback: {feedback}
    """   
    
def run_app():
    if 'page' not in st.session_state:
        st.session_state.page = "main"

    st.title("Personalized Recommender System")

    if 'user_details' not in st.session_state:
        st.session_state.user_details = None

    if 'models' not in st.session_state:
        st.session_state.models = initialize_models()

    # Collect user details
    collect_user_details()

    # Unpack models
    llama_model, vector_store, phi_model = st.session_state.models

    # Handle user input
    user_input, submitted = handle_user_input()

    # Process user input and generate response
    if submitted and user_input:
        user_details = st.session_state.user_details
            
        PersonalisedJson=personalize_analyzer(user_input, phi_model)
        if(PersonalisedJson['ispersonalized']=="yes"):
            personalized_context = PersonalisedJson['personalizedcontent']
            
            # Load or create the vector store
            vector_store.add_content_to_vector_store(personalized_context)
        
        queries = multiQueryRetreiver(user_input, phi_model)
        context = docRetreiver(vector_store, queries)
        
        response = user_input
        
        if len(context)!=0:
            feedback = ""
            for i in range(RETRY):
                response, score, feedback = try_generate(user_input, context, phi_model, llama_model, feedback)
                if len(feedback)!=0:
                    feedback = formatFeedback(response, feedback)
                
                if score > EVAL_THRESHOLD:
                    break
        
        st.write(response)
                        
    display_chat()