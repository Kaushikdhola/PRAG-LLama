import streamlit as st

def display_chat():
    # for chat in st.session_state.chat_history:
    #     if 'user' in chat:
    #         st.text(f"You: {chat['user']}")
    #     if 'bot' in chat:
    #         st.text(f"Bot: {chat['bot']}")
    pass

def handle_user_input():
    return st.text_input("You: ", key="input")
