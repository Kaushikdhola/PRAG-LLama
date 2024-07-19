import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings

def main():
    st.title("Simple Embedding Test")
    
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    user_input = st.text_input("Enter a sentence to embed:")
    
    if user_input:
        embedding = embedding_model.embed_query(user_input)
        st.write(f"Embedding generated. Shape: {len(embedding)}")
        st.write(embedding[:10])  # Display first 10 elements

if __name__ == "__main__":
    main()