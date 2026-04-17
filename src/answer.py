from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

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

1. Short Answer
[Your short, concise answer here]

2. Source-Backed Explanation
[Detailed explanation based strictly on the chunks]

3. Sources Used:
[List each used source as: * document_name + chunk_id]

4. Confidence Level:
[High / Medium / Low]

5. Abstention:
[If applicable, state the abstention message, otherwise leave blank or write N/A]
"""

def generate_answer(query: str, retrieved_chunks: list):
    # Rule: If no chunks retrieved, strict abstention
    if not retrieved_chunks:
        return "Unable to confirm from approved documents. Manual review required."
        
    # We use Flash 2.5 with zero temperature to prevent hallucinations
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
    
    formatted_chunks = []
    for chunk in retrieved_chunks:
        # Pass document names and chunk_ids for valid attribution
        doc_name = chunk.metadata.get("document_name", "Unknown-Doc")
        chunk_id = chunk.metadata.get("chunk_id", "Unknown-ID")
        text = chunk.page_content
        formatted_chunks.append(f"--- SOURCE: {doc_name} + {chunk_id} ---\n{text}\n")
        
    context = "\n".join(formatted_chunks)
    
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm
    
    response = chain.invoke({"context": context, "query": query})
    return response.content

def process_query(query: str, country: str, category: str):
    """Orchestrates retrieval and generation gracefully."""
    from src.retrieve import retrieve_chunks
    try:
        # Retrieve optimal number of chunks (between 3 to 5)
        chunks = retrieve_chunks(query, country, category, top_k=4)
        answer = generate_answer(query, chunks)
        return answer, chunks
    except Exception as e:
        return f"System Error: {str(e)}", []
