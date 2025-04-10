from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.vectorstores import FAISS  # FAISS integration (commented out)
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import pinecone
from pinecone import Pinecone, ServerlessSpec
import os

# Pinecone settings
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_ENV = "us-east-1"  # e.g., "us-east-1"
INDEX_NAME = "test"  # Your desired index name

# Explicitly set the API key in the environment
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

def create_embeddings():
    try:
        # Load PDF files from the "Data" directory
        loader = PyPDFDirectoryLoader("Data")
        documents = loader.load()

        # Combine all documents into a single text block
        full_text = "\n".join([doc.page_content for doc in documents])

        # Split text into chunks for better context retention
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,  # Increase chunk size for better context retention
            chunk_overlap=100
        )
        chunks = text_splitter.split_text(full_text)

        # Define embedding model
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs = {"device": device}
        embeddings_hf = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

        # ----------------------
        # Pinecone integration using the new instance-based API
        # ----------------------
        # Create a Pinecone instance using the API key from the environment
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        
        # Check if the index exists; if not, create it with dimension 384 (for our embedding model)
        if INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=INDEX_NAME,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
            )
            print(f"✅ Pinecone index '{INDEX_NAME}' created.")
        else:
            print(f"✅ Pinecone index '{INDEX_NAME}' already exists.")

        # Import Pinecone vectorstore from the updated module to avoid deprecation warnings
        from langchain_community.vectorstores import Pinecone as LC_Pinecone

        # Create the vectorstore in Pinecone using the text chunks and embeddings
        vectorstore = LC_Pinecone.from_texts(chunks, embeddings_hf, index_name=INDEX_NAME)
        print("✅ Vectors successfully created and saved in Pinecone!")
    
    except Exception as e:
        print(f"❌ Error creating vectors: {e}")

if __name__ == "__main__":
    create_embeddings()
