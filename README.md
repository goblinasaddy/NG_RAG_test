# nitarya_rag_v1

A production-quality Retrieval-Augmented Generation (RAG) system specifically designed for internal research use cases.
This system strictly answers ONLY from approved internal documents and avoids hallucinations by automatically abstaining when evidence is missing.

## Goal
Build a safe, reliable, and evidence-driven system that acts as a research assistant, without exposing external knowledge or hallucinated statements.

## Tech Stack
- Python
- LangChain
- FAISS (Local Vector Database)
- Google Gemini API (`models/embedding-001` & `gemini-2.5-flash`)
- Streamlit

## Folder Structure
```text
nitarya_rag_v1/
│
├── data/approved_docs/      # Place your raw .json documents here
├── processed/
│   ├── clean_docs/          # Processed & approved documents
│   ├── chunks/              # Chunked text data
│   └── embeddings/          # FAISS local vector store
├── src/
│   ├── ingest.py            # Filters and approves raw documents
│   ├── chunk.py             # Chunks documents using RecursiveCharacterTextSplitter
│   ├── embed.py             # Creates FAISS embeddings
│   ├── retrieve.py          # Strict similarity search with metadata filtering
│   ├── answer.py            # Strict LLM generation with abstention rules
│   ├── app.py               # CLI interface
│   └── streamlit_app.py     # Streamlit UI
│
├── tests/eval_queries.md    # 20 evaluation queries
├── requirements.txt         # Dependencies
└── README.md
```

## Setup Instructions

1. **Install Dependencies**
   Navigate to the target folder and install packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**
   Create a `.env` file in the root directory (`nitarya_rag_v1/`) and add your Google Gemini API Key:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

3. **Data Preparation**
   Place your approved internal documents in `data/approved_docs/` as `.json` files.
   Each file MUST strictly match this schema in order to route and inject effectively:
   ```json
   {
       "text": "Your document content goes here...",
       "metadata": {
           "document_name": "InvestmentLaw2023",
           "country": "UAE",
           "category": "Ownership rules",
           "source_type": "PDF",
           "version": "1.0",
           "approval_status": "approved"
       }
   }
   ```
   
   **Excel Policy Logs (.xlsx)**
   You can natively place policy log trackers (`.xlsx` files) anywhere inside the `data/` folder! These files must include columns such as `Year`, `What changed`, `Area Affected`, `Why it matters`, and `Source`. The ingestion pipeline reads them dynamically, structures each row into a distinct document chunk, and assigns it standard "Policy" metadata dynamically! There is no need for manual JSON structuring for policy logs.

4. **Run the Ingestion Pipeline**
   ```bash
   python -m src.ingest
   python -m src.chunk
   python -m src.embed
   ```

## Running the System

### via CLI
Start the interactive command-line interface from the root directory:
```bash
python -m src.app
```

### via Streamlit UI
Start the interactive UI dashboard:
```bash
streamlit run src/streamlit_app.py
```

## System Rules & Flow
1. **Ingest**: Checks `approval_status == 'approved'`. Drops invalid files.
2. **Chunk**: Uses token splits ranging 500-800 lengths with overlap to maintain text logics. 
3. **Retrieve**: Strictly applies both country and category filters so responses don't mix jurisdictions.
4. **Answer**: Controlled generation prompts the LLM to output exactly the structured form or cleanly abstain by saying: *"Unable to confirm from approved documents. Manual review required."*
