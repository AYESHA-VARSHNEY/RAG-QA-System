import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

class RAGEngine:
    def __init__(self):
        self.dimension = 384
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []

    def process_document(self, file_path, ext):
        text = ""
        if ext == ".pdf":
            reader = PdfReader(file_path)
            for page in reader.pages:
                text += page.extract_text()
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

        # Chunking Strategy: 500 chars with 50 overlap
        new_chunks = [text[i:i+500] for i in range(0, len(text), 450)]
        self.chunks.extend(new_chunks)
        
        embeddings = model.encode(new_chunks)
        self.index.add(np.array(embeddings).astype('float32'))
        return len(new_chunks)

    def retrieve(self, query):
        query_vec = model.encode([query])
        D, I = self.index.search(np.array(query_vec).astype('float32'), k=3)
        return [self.chunks[i] for i in I[0] if i != -1]