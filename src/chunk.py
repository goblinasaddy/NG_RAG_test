import json
import os
import glob
import uuid
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # Ensure backward compatibility if someone uses an older pip packages layout
    from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_documents(input_dir="processed/clean_docs", output_dir="processed/chunks"):
    """
    Reads approved documents.
    Chunks text directly while maintaining logical meaning using RecursiveCharacterTextSplitter.
    Output size: 500-800 tokens, 80-120 overlap.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # We use RecursiveCharacterTextSplitter with the tiktoken encoder to match token size limits
    # chunk_size between 500-800, overlap between 80-120.
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=750, 
        chunk_overlap=100
    )
    
    file_pattern = os.path.join(input_dir, "*.json")
    for file_path in glob.glob(file_pattern):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                text = data.get("text", "")
                metadata = data.get("metadata", {})
                
                # Do not break logical meaning: split the document correctly into chunks
                chunks = splitter.split_text(text)
                
                for chunk_text in chunks:
                    chunk_id = f"{metadata.get('document_name', 'doc')}_{uuid.uuid4().hex[:8]}"
                    
                    chunk_data = {
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                        "metadata": metadata
                    }
                    
                    output_path = os.path.join(output_dir, f"{chunk_id}.json")
                    with open(output_path, "w", encoding="utf-8") as out_f:
                        json.dump(chunk_data, out_f, indent=2)
            except Exception as e:
                print(f"Error chunking {file_path}: {e}")
                    
    print(f"Chunking complete. Chunks saved to {output_dir}")

if __name__ == "__main__":
    print("Starting chunking process...")
    chunk_documents()
