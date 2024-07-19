import streamlit as st
import json
import os

USER_DETAILS_FILE = "user_details.json"

def load_user_details():
    if os.path.exists(USER_DETAILS_FILE):
        with open(USER_DETAILS_FILE, "r") as f:
            return json.load(f)
    return None

def save_user_details(details):
    with open(USER_DETAILS_FILE, "w") as f:
        json.dump(details, f)

def collect_user_details():
    # Load user details from file if they exist
    if st.session_state.user_details is None:
        st.session_state.user_details = load_user_details()
    
    if st.session_state.user_details is None:
        st.subheader("Please provide your details")
        age = st.number_input("Age", min_value=1)
        profession = st.text_input("Profession")
        qualification = st.text_input("Qualification")
        
        if st.button("Submit"):
            st.session_state.user_details = {
                "age": age,
                "profession": profession,
                "qualification": qualification
            }
            save_user_details(st.session_state.user_details)
            st.experimental_rerun()
    else:
        st.write("Welcome back!")
        st.write(f"Age: {st.session_state.user_details['age']}")
        st.write(f"Profession: {st.session_state.user_details['profession']}")
        st.write(f"Qualification: {st.session_state.user_details['qualification']}")
