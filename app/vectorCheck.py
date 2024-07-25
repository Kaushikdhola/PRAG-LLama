import os
from langchain_community.vectorstores import FAISS


def ensure_directory_exists(directory_path):
    """Ensure that the specified directory exists."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def load_or_create_vector_store(directory_path, embeddings):
    """Load existing vector store or create a new one if it doesn't exist."""
    if os.path.exists(os.path.join(directory_path, 'index.faiss')):
        return FAISS.load_local(directory_path, embeddings, allow_dangerous_deserialization=True)
    else:
        # Create an empty FAISS index
        return FAISS.from_texts(["dummy"], embeddings)  # Use a dummy text to create the index

def add_content_to_vector_store(vector_store, content, embeddings):
    """Add new content to the vector store."""
    for item in content:
        vector_store.add_texts([item], embeddings=[embeddings.embed_query(item)])

def save_vector_store(vector_store, directory_path):
    """Save the vector store to the specified directory."""
    vector_store.save_local(directory_path)
    
    if hasattr(vector_store, 'get_all_documents'):
        documents = vector_store.get_all_documents()
        for i, doc in enumerate(documents, 1):
            print(f"Chunk {i}:")
            print(doc.page_content)
            print("-" * 50)

def retriever(query, vector_store):
    """Retrieve the most similar content from the vector store."""
    return vector_store.as_retriever().invoke(query)
