import os
import faiss
import numpy as np
import pandas as pd
from langchain.document_loaders import UnstructuredPDFLoader
from flagai.embeddings import BgeEmbedding

# Install necessary libraries if not already installed
# !pip install langchain pandas faiss-cpu flagai

# Helper Functions
def load_pdf(file_path):
    """Uses LangChain's UnstructuredPDFLoader to extract text and metadata from a PDF file."""
    loader = UnstructuredPDFLoader(file_path)
    documents = loader.load()
    text = "\n".join([doc.page_content for doc in documents])
    metadata = [doc.metadata for doc in documents]
    return text, metadata

def load_excel(file_path):
    """Extracts text from an Excel file."""
    df = pd.read_excel(file_path)
    return "\n".join([" ".join(map(str, row)) for row in df.values])

def split_into_chunks(text, max_length=512):
    """Splits large text into smaller chunks for embedding."""
    words = text.split()
    return [" ".join(words[i:i + max_length]) for i in range(0, len(words), max_length)]

# Load Flag Embeddings (BGE-M3)
embedding_model = BgeEmbedding("BAAI/bge-base-en")


def embed_texts(texts):
    """Generates embeddings for a list of texts using FlagEmbedding BGE-M3."""
    embeddings = embedding_model.embed(texts)
    return np.array(embeddings)

# Initialize FAISS Index
embedding_dim = 768  # Dimension of BGE-M3 embeddings
index = faiss.IndexFlatL2(embedding_dim)  # L2 similarity for FAISS
metadata_store = []


def store_in_faiss(embeddings, metadata):
    """Stores embeddings and metadata in FAISS."""
    index.add(embeddings)
    metadata_store.extend(metadata)

def retrieve(query, top_k=5):
    """Retrieves top-k similar documents for a given query."""
    query_embedding = embed_texts([query])
    distances, indices = index.search(query_embedding, top_k)
    results = [(metadata_store[i], distances[0][idx]) for idx, i in enumerate(indices[0])]
    return results

# Main Workflow
file_path = "/path/to/file"
if file_path.endswith(".pdf"):
    document_text, metadata = load_pdf(file_path)
    chunks = split_into_chunks(document_text)
elif file_path.endswith(".xlsx"):
    document_text = load_excel(file_path)
    chunks = split_into_chunks(document_text)
    metadata = [{} for _ in chunks]  # Empty metadata for Excel
else:
    raise ValueError("Unsupported file type. Please use a PDF or Excel file.")

# Embed chunks
print("Generating embeddings...")
embeddings = embed_texts(chunks)

# Store in FAISS
store_in_faiss(embeddings, metadata)
print("Embeddings and metadata stored in FAISS.")

# Query for retrieval
query = input("Enter your query: ")
results = retrieve(query)
print("Top results:")
for i, (meta, distance) in enumerate(results, 1):
    print(f"{i}. Metadata: {meta}, Distance: {distance}")
