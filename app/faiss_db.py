import faiss
import numpy as np

class FaissDB:
    def __init__(self, dimension=768):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
    
    def store_vector(self, vector):
        vector_np = np.array(vector).astype('float32').reshape(1, -1)
        self.index.add(vector_np)
    
    def search(self, vector, k=1):
        vector_np = np.array(vector).astype('float32').reshape(1, -1)
        distances, indices = self.index.search(vector_np, k)
        return distances, indices
