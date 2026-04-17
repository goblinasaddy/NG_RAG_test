import sys
from unittest.mock import MagicMock

# Aggressively mock transformers and its submodules to prevent the numpy version check crash
mock_transformers = MagicMock()
sys.modules["transformers"] = mock_transformers
sys.modules["transformers.utils"] = MagicMock()
sys.modules["transformers.utils.versions"] = MagicMock()
sys.modules["transformers.dependency_versions_check"] = MagicMock()

import json
import os
import glob
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load env variables including GOOGLE_API_KEY
load_dotenv()

def embed_chunks(input_dir="processed/chunks", output_dir="processed/embeddings"):
    """
    Reads chunks and converts text into embeddings using Gemini API.
    Saves FAISS index locally holding metadata and chunk mapping.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Use Gemini Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    docs = []
    file_pattern = os.path.join(input_dir, "*.json")
    for file_path in glob.glob(file_pattern):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                text = data.get("text", "")
                metadata = data.get("metadata", {})
                
                # Merge chunk_id to metadata explicitly
                metadata["chunk_id"] = data.get("chunk_id", "")
                
                doc = Document(page_content=text, metadata=metadata)
                docs.append(doc)
            except Exception as e:
                print(f"Error embedding {file_path}: {e}")
            
    if not docs:
        print("No chunks found to embed.")
        return
        
    print(f"Embedding {len(docs)} chunks...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # Save index locally
    vectorstore.save_local(output_dir)
    print(f"FAISS index saved to {output_dir}")

if __name__ == "__main__":
    embed_chunks()
