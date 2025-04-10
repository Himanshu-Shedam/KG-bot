from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_community.vectorstores import FAISS  # FAISS integration (commented out)
from dotenv import load_dotenv
import os
import re
import torch
import pinecone  # Import Pinecone

load_dotenv()
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Pinecone settings
# NOTE: Ensure your API key string does not include extra text.
# PINECONE_API_KEY = "pcsk_3poQiw_2RF93mAiUBwTzPEntB64e7SkGxzvuwhKcuXKithkZKQo1nFa77PvYRoLsSrEvxg"
PINECONE_ENV = "us-east-1"  # Replace with your Pinecone environment if different
INDEX_NAME = "test"  # Your Pinecone index name
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

def clean_response(response_text):
    if not response_text:
        return "No response was generated."
    clean_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
    return clean_text

def get_response(text):
    # Define embedding model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"device": device}
    embeddings_hf = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    
    # Create a Pinecone instance using the API key from the environment
    from pinecone import Pinecone, ServerlessSpec
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    
    # NOTE: We assume that the index already exists. If not, create it externally or via separate code.
    # Import Pinecone vectorstore from the updated module to avoid deprecation warnings
    from langchain_community.vectorstores import Pinecone as LC_Pinecone
    vectorstore = LC_Pinecone.from_existing_index(index_name=INDEX_NAME, embedding=embeddings_hf)
    
    # Set up the LLM via ChatGroq
    # llm = ChatGroq(
    #     temperature=1, 
    #     model_name="llama-3.3-70b-versatile", 
    #     groq_api_key=GROQ_API_KEY
    # )
    # Alternative LLM via ChatGoogleGenerativeAI (commented out)
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")
    
    rag_template = """
You are an AI-powered assistant providing actionable solutions and detailed information for queries related to the provided data. Your task is to provide responses based only on retrieved documents without adding extra steps or extraneous information.

Guidelines:
1. **Data-Related Queries:**
   - If the retrieved document contains step-by-step instructions, reproduce **only the listed steps exactly as they appear, without adding titles**.
   - Do not introduce any new steps, even if you think they are helpful.
   - If the document contains general information without steps, summarize concisely and rephrase it in clear, understandable language.
   - **If no relevant context is found (i.e., the Context from Retrieved Documents is empty), respond with:**
     "I don't know the answer based on the available context." Do not generate any further response.
   - If the answer includes a link, append the following line at the end: "for more details visit this SOP link: [sop_link]" (using the provided link, without square brackets). If no link is provided, do not append any extra text.

2. **Conversational Queries:**
   - Handle conversational queries (e.g., greetings, thanks, or small talk) naturally, without referencing the document context.
   - Keep responses friendly, concise, and engaging.

3. **Mixed Queries:**
   - Address both informational and conversational parts separately, ensuring clarity in the response.

Dialogue Flow Guidelines:
- Begin responses with a human-like tone.
- Format steps as numbered points without additional titles or commentary.
- Avoid phrases like "Based on the documents..." or speculative statements.
- Provide point-wise, easy-to-understand instructions.
- Only append the link line ("for more details visit this SOP link: [sop_link]") if a valid link is provided in the retrieved context.
- answer only based on retreved data dont make up answers.

Context from Retrieved Documents:
{context}

User Query:
{question}

Your Response:
"""


    rag_prompt = ChatPromptTemplate.from_template(rag_template)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm, 
        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}), 
        chain_type_kwargs={"prompt": rag_prompt},
    )
    
    response = qa_chain.invoke(text)
    # Optionally clean the response if needed
    # response = clean_response(response['result'])
    return response['result']

def get_gemini_response(input, image):
    model = genai.GenerativeModel('gemini-2.0-flash-lite')
    response = model.generate_content([input, image])
    print(response.text)
    return response.text
