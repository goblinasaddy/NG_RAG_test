import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from src.answer import process_query

st.set_page_config(page_title="Internal Research RAG", page_icon="🔒", layout="wide")

st.title("Internal Research RAG")
st.subheader("Controlled, evidence-based retrieval system")
st.markdown("⚠️ **Notice:** This system strictly uses approved documents. If there is insufficient evidence, it will abstain.")

# Define available constraints
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
    st.header("Search Filters")
    country = st.selectbox("Select Country", COUNTRIES)
    category = st.selectbox("Select Category", CATEGORIES)
    st.info("Strict filtering prevents the AI from mixing context from different countries or categories.")

query = st.text_area("Enter your query:", height=100)

if st.button("Get Answer", type="primary"):
    if not query.strip():
        st.warning("Please enter a query to proceed.")
    else:
        with st.spinner("Retrieving approved documents and analyzing..."):
            answer, chunks = process_query(query, country, category)
            
            st.markdown("---")
            st.markdown("### Answer")
            
            # Catch Abstention
            if answer.strip() == "Unable to confirm from approved documents. Manual review required.":
                st.error("🔒 **" + answer + "**")
            else:
                st.markdown(answer)
                
            st.markdown("---")
            with st.expander("🔍 View Retrieved Sources (Debug)"):
                if not chunks:
                    st.info("No chunks were retrieved from the vector store.")
                for i, chunk in enumerate(chunks):
                    # Safely load metadata
                    c_id = chunk.metadata.get('chunk_id', 'Unknown')
                    d_name = chunk.metadata.get('document_name', 'Unknown')
                    c_country = chunk.metadata.get('country', 'Unknown')
                    
                    st.markdown(f"**Source {i+1}** - ID: `{c_id}` | Doc: `{d_name}` | Country: `{c_country}`")
                    st.caption("Content:")
                    st.text(chunk.page_content)
                    st.markdown("---")
