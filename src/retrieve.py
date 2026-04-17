import sys
from unittest.mock import MagicMock

# Aggressively mock transformers and its submodules to prevent the numpy version check crash
mock_transformers = MagicMock()
sys.modules["transformers"] = mock_transformers
sys.modules["transformers.utils"] = MagicMock()
sys.modules["transformers.utils.versions"] = MagicMock()
sys.modules["transformers.dependency_versions_check"] = MagicMock()

import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def get_retriever():
    """Load FAISS index and return vectorstore object."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    index_path = "processed/embeddings"
    
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}. Please run embed.py first.")
        
    # Local loading with dangerous deserialization is safe since it's locally generated
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

def retrieve_chunks(query: str, country: str, category: str, top_k: int = 4):
    """
    Retrieve top_k chunks for a query with STRICT filtering applied first.
    Filters by country and category explicitly preventing mixed results.
    """
    vectorstore = get_retriever()
    
    # Apply strict filtering before/during retrieval
    filter_dict = {
        "country": country,
        "category": category
    }
    
    # similarity_search supports dict filter natively in local FAISS for exact matches
    results = vectorstore.similarity_search(query, k=top_k, filter=filter_dict)
    return results

if __name__ == "__main__":
    # Test script run
    try:
        res = retrieve_chunks("What are the ownership rules?", "UAE", "Ownership rules")
        print(f"Retrieved {len(res)} chunks.")
        for r in res:
            print(r.metadata.get("chunk_id"), r.page_content[:50])
    except Exception as e:
        print("Could not test retrieval:", e)
