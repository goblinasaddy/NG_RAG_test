import json
import os
import glob
import uuid
try:
    import pandas as pd
except ImportError:
    pass
from langchain_core.documents import Document

def process_excel_policy_log(file_path, output_chunk_dir="processed/chunks"):
    """
    Reads an Excel file and converts each row into a structured LangChain Document.
    Saves each Document immediately to processed/chunks as it should not be chunked further.
    Returns a list of LangChain Document objects.
    """
    os.makedirs(output_chunk_dir, exist_ok=True)
    
    filename_lower = os.path.basename(file_path).lower()
    country = "Unknown"
    if "uae" in filename_lower:
        country = "UAE"
    elif "australia" in filename_lower:
        country = "Australia"
    elif "thailand" in filename_lower:
        country = "Thailand"
        
    documents = []
    
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error reading Excel {file_path}: {e}")
        return documents
        
    for idx, row in df.iterrows():
        # Handle nan values gracefully
        row_dict = {str(k).strip(): ("" if pd.isna(v) else str(v).strip()) for k, v in row.items()}
        
        # Mapping properties
        year = row_dict.get("Year", "")
        what_changed = row_dict.get("What changed", "")
        area_affected = row_dict.get("Area Affected", "")
        why_it_matters = row_dict.get("Why it matters", "")
        source = row_dict.get("Source", "")
        
        text_content = f"Country: {country}\nCategory: Policy changes\n\nPolicy Type: {area_affected}\nPolicy Change: {what_changed}\nEffective Date: {year}\n\nSource: {source}\n\nNotes: {why_it_matters}"
        
        unique_id = uuid.uuid4().hex[:8]
        row_id = f"policy_log_{unique_id}"
        
        metadata = {
            "document_name": "Policy_Log",
            "country": country,
            "category": "Policy changes",
            "source_type": "Internal Spreadsheet",
            "version": "v1",
            "approval_status": "approved",
            "row_id": row_id,
            "chunk_id": row_id
        }
        
        doc = Document(page_content=text_content, metadata=metadata)
        documents.append(doc)
        
        # Save straight to chunks directory! DO NOT CHUNK FURTHER.
        chunk_data = {
            "chunk_id": row_id,
            "text": text_content,
            "metadata": metadata
        }
        chunk_path = os.path.join(output_chunk_dir, f"{row_id}.json")
        with open(chunk_path, "w", encoding="utf-8") as out_f:
            json.dump(chunk_data, out_f, indent=2)
            
    print(f"Processed Excel file {os.path.basename(file_path)}: Generated {len(documents)} structured chunks.")
    return documents

def ingest_documents(input_dir="data/approved_docs", output_dir="processed/clean_docs"):
    """
    Reads JSON documents and Excel logs from the data directory.
    Rejects JSON documents where approval_status != 'approved'.
    Saves clean JSON to clean_docs.
    Returns clean JSON chunks and Excel LangChain Document objects.
    """
    os.makedirs(output_dir, exist_ok=True)
    clean_docs = []
    
    # Process JSON files
    json_pattern = os.path.join(input_dir, "*.json")
    for file_path in glob.glob(json_pattern):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                metadata = data.get("metadata", {})
                
                # Check strict approval field
                if metadata.get("approval_status") == "approved":
                    output_path = os.path.join(output_dir, os.path.basename(file_path))
                    with open(output_path, "w", encoding="utf-8") as out_f:
                        json.dump(data, out_f, indent=2)
                    clean_docs.append(output_path)
                    print(f"Ingested and approved JSON: {file_path}")
                else:
                    print(f"Rejected JSON: {file_path} (not approved)")
            except Exception as e:
                print(f"Error processing JSON {file_path}: {e}")
                
    # Process Excel files looking dynamically in the general data folder structure recursively
    excel_docs = []
    excel_pattern = os.path.join("data", "**", "*.xlsx")
    for file_path in glob.glob(excel_pattern, recursive=True):
        docs = process_excel_policy_log(file_path)
        excel_docs.extend(docs)
        
    return clean_docs, excel_docs

if __name__ == "__main__":
    print("Starting ingestion process...")
    json_docs, xlsx_docs = ingest_documents()
    print("Ingestion complete.")
