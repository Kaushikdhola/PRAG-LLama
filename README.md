# Personalized Recommender System

## Overview

This project is a personalized recommender system that uses advanced natural language processing techniques to provide tailored recommendations to users. It combines a chat interface with user profiling and context-aware response generation.

## Features

- **Interactive Chat Interface**: Built with Streamlit for a user-friendly experience.
- **User Profiling**: Collects and utilizes user details for personalized interactions.
- **Context-Aware Responses**: Generates responses based on user input, previous interactions, and user profile.
- **RAG (Retrieval-Augmented Generation)**: Enhances responses with relevant information from a knowledge base.
- **Similarity Matching**: Uses semantic similarity to determine the relevance of previous responses.

## Technologies Used

- Python
- Streamlit
- Langchain
- LLaMa (Language Model)
- FAISS (Vector Store)
- HuggingFace Transformers
- Sentence Transformers

## Setup and Installation

1. Clone the repository:
```
git clone https://github.com/Kaushikdhola/PRAG-LLama.git
```
```
cd PRAG-LLama
```
2. Install the required dependencies:
```
pip install -r requirements.txt
```
3. Set up the LLaMa model (follow LLaMa's official instructions for downloading and setting up the model).

4. Run the Streamlit app:

```
streamlit run app/main.py
```


## Project Structure

- `streamlit_app.py`: The main Streamlit application file.
- `app/`
- `main.py`: central point of execution of all the logic.
- `user_details.py`: Handles user profile collection.
- `chat.py`: Manages the chat interface and user input handling.
- `response_generation.py`: Contains the core logic for generating personalized responses.
- `faiss_db.py`: Manages the FAISS vector database.

## How It Works

1. **User Profiling**: On first use, the system collects user details such as age, profession, and qualifications.
2. **Chat Interface**: Users interact with the system through a chat-like interface.
3. **Response Generation**: 
- The system uses the LLaMa model to generate responses.
- It considers the user's profile, current input, and context from previous interactions.
- Relevant information is retrieved from the FAISS vector store if the current query is similar to previous interactions.
4. **Continuous Learning**: Each interaction is stored and used to improve future recommendations.
