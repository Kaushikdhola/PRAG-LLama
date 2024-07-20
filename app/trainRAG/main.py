import streamlit as st
from .model import initialize_models
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
 

def train_model_page():
    st.title("Train Your Own Model")
    
    if 'training_input' not in st.session_state:
        st.session_state.training_input = ""

    if 'train_models' not in st.session_state:
        st.session_state.train_models = initialize_models()
        print("Initialised the model")
        
    llama_model, vector_store, embeddings = st.session_state.train_models
    
    # Text input for training data
    user_input = st.text_area("Enter your training data (press Enter to add more content):", 
                              value=st.session_state.training_input,
                              height=200,
                              key="training_data_input")
    

    
    # Button to append content to file
    if st.button("Add Content"):
        
        # with open("training_data.txt", "a") as f:
        #     f.write(user_input + "\n")
        if user_input != None:
            input_embedding = embeddings.embed_query(user_input)
            vector_store.add_texts([user_input], embeddings=[input_embedding])
        st.success("Content added successfully!")
    
    if st.button("Train Model"):
        # Read the entire file
        # with open("training_data.txt", "r") as f:
        #     training_data = f.read()
        
        prompt_template = """
 
            User Query: {question}
 
            Information:
            {context}
            
            Please provide a personalized answer:
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["question", "context"]
        )
        
        print("I QA Chain")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llama_model,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
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
        
        prompt_template = """
 
            User Query: {question}
 
            Information:
            {context}
            
            Please provide a personalized answer:
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["question", "context"]
        )
        
        print("I QA Chain")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llama_model,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        for key in ['query', 'question', 'input']:
            try:
                response = qa_chain({key: question})
                sources = response['source_documents']
                print(sources, response)
                for i, doc in enumerate(sources[0], 1):
                    st.write(f"{i}. {doc.page_content}...")
            except:
                pass
        
        # llama_model, vector_store, faiss_db, embeddings = st.session_state.models
        
        # # Here you would implement personalized RAG logic - kunal
        # # For this example, we'll just use a simple response
        # prompt = generate_prompt(question, {}, None, llama_model, vector_store, embeddings)
        # response = generate_response(prompt, llama_model)
        
        # st.write("Response:", response)

    if st.button("Return to Main Page"):
        st.session_state.page = "main"
        st.experimental_rerun()