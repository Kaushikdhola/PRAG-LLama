import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DIRECTORY_PATH = './personalized'

class VectorStore:
    def __init__(self, directory_path=DIRECTORY_PATH):
        self.embeddings = HuggingFaceEmbeddings()
        self.store = self._load_or_create_vector_store(self.embeddings)
            
    def _load_or_create_vector_store(self, embeddings):
        """Load existing vector store or create a new one if it doesn't exist."""
        if os.path.exists(os.path.join(DIRECTORY_PATH, 'index.faiss')):
            return FAISS.load_local(DIRECTORY_PATH, embeddings, allow_dangerous_deserialization=True)
        else:
            # Create an empty FAISS index
            return FAISS.from_texts([""], embeddings) 
    
    def add_content_to_vector_store(self,content):
        """Add new content to the vector store."""
        for item in content:
            self.store.add_texts([item], embeddings=[self.embeddings.embed_query(item)])
        
        self._save_vector_store()
        
    def _save_vector_store(self):
        """Save the vector store to the specified directory."""
        self.store.save_local(DIRECTORY_PATH)
    
    def search(self, query):
        return self.store.similarity_search_with_relevance_scores(query)
