import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Nitarya Internal RAG System", page_icon="🔒", layout="wide")

PROMPT_TEMPLATE = """You are a strictly controlled, evidence-based research assistant.
Your task is to answer the user's query ONLY using the provided retrieved chunks.

CRITICAL RULES:
1. NEVER hallucinate.
2. ALWAYS provide source attribution.
3. NEVER assume or use external/prior knowledge.
4. If there are no relevant chunks, conflicting data, or weak evidence, you MUST abstain.

ABSTENTION RULE:
If you cannot fully answer the question based strictly on the provided chunks, you must return EXACTLY the following text and nothing else:
"Unable to confirm from approved documents. Manual review required."

Provided Chunks:
{context}

User Query:
{query}

You must return the response EXACTLY in the following format:

### Short Answer
[Your short, concise answer here]

### Source-Backed Explanation
[Detailed explanation based strictly on the chunks]

### Sources Used
* [document_name] + [chunk_id]

### Confidence Level
[High / Medium / Low]

### Abstention (if any)
[If applicable, state the abstention message, otherwise leave blank or write N/A]
"""

@st.cache_resource(show_spinner="Loading FAISS index...")
def get_retriever(api_key: str):
    """Load FAISS index and return vectorstore object."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
    
    # Resolve the path to the FAISS index relative to this script
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_path = os.path.join(base_dir, "processed", "embeddings")
    
    # Fallback to current working directory if not found
    if not os.path.exists(index_path):
        index_path = "processed/embeddings"
        
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}. Please ensure processed/embeddings exists.")
        
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

def retrieve_chunks(vectorstore, query: str, country: str, category: str, top_k: int = 4):
    filter_dict = {
        "country": country,
        "category": category
    }
    results = vectorstore.similarity_search(query, k=top_k, filter=filter_dict)
    return results

def generate_answer(query: str, retrieved_chunks: list, api_key: str):
    if not retrieved_chunks:
        return "Unable to confirm from approved documents. Manual review required."
        
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0, google_api_key=api_key)
    
    formatted_chunks = []
    for chunk in retrieved_chunks:
        doc_name = chunk.metadata.get("document_name", "Unknown-Doc")
        chunk_id = chunk.metadata.get("chunk_id", "Unknown-ID")
        text = chunk.page_content
        formatted_chunks.append(f"--- SOURCE: {doc_name} + {chunk_id} ---\n{text}\n")
        
    context = "\n".join(formatted_chunks)
    
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm
    
    response = chain.invoke({"context": context, "query": query})
    return response.content

st.title("Nitarya Internal RAG System")
st.subheader("Controlled, evidence-based retrieval system")
st.markdown("⚠️ **Notice:** This system strictly uses approved documents. If there is insufficient evidence, it will abstain.")

COUNTRIES = ["UAE", "Australia", "Thailand"]
CATEGORIES = [
    "Ownership rules",
    "Tax exposure",
    "Investor visa / residency",
    "Capital controls",
    "Estate / inheritance tax",
    "Policy changes"
]

with st.sidebar:
    st.header("Authentication")
    api_key = st.text_input("Enter Gemini API Key", type="password", key="gemini_api_key_input")
    
    st.header("Search Filters")
    country = st.selectbox("Select Country", COUNTRIES)
    category = st.selectbox("Select Category", CATEGORIES)
    st.info("Strict filtering prevents the AI from mixing context from different countries or categories.")

query = st.text_area("Enter your query:", height=100)

if st.button("Get Answer", type="primary"):
    if not api_key:
        st.warning("⚠️ Please enter your Gemini API Key in the sidebar to proceed.")
    elif not query.strip():
        st.warning("Please enter a query to proceed.")
    else:
        try:
            with st.spinner("Initializing system and retrieving approved documents..."):
                vectorstore = get_retriever(api_key)
                chunks = retrieve_chunks(vectorstore, query, country, category, top_k=4)
                
            with st.spinner("Analyzing documents and generating answer..."):
                answer = generate_answer(query, chunks, api_key)
                
            st.markdown("---")
            
            if answer.strip() == "Unable to confirm from approved documents. Manual review required.":
                st.error("🔒 **" + answer + "**")
            else:
                st.markdown(answer)
                
            st.markdown("---")
            with st.expander("🔍 View Retrieved Sources (Debug)"):
                if not chunks:
                    st.info("No chunks were retrieved from the vector store.")
                for i, chunk in enumerate(chunks):
                    c_id = chunk.metadata.get('chunk_id', 'Unknown')
                    d_name = chunk.metadata.get('document_name', 'Unknown')
                    c_country = chunk.metadata.get('country', 'Unknown')
                    
                    st.markdown(f"**Source {i+1}** - ID: `{c_id}` | Doc: `{d_name}` | Country: `{c_country}`")
                    st.caption("Content:")
                    st.text(chunk.page_content)
                    st.markdown("---")
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
