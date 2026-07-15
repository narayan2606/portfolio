import os
import json
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer

# Load credentials from .env file
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") 

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("CRITICAL ERROR: Supabase credentials not found in .env file.")

print("[System] Initializing Supabase client...")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print("[System] Loading local embedding model (all-MiniLM-L6-v2)...")
print("[System] This might take a few seconds on the first run as it downloads the model weights (~80MB).")
# This model generates exactly 384 dimensions, matching our PostgreSQL vector(384) schema
encoder = SentenceTransformer('all-MiniLM-L6-v2')

def load_portfolio_data():
    try:
        with open('portfolio_data.json', 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError("CRITICAL ERROR: 'portfolio_data.json' not found in the current directory.")

def inject_data_to_db():
    data = load_portfolio_data()
    total_chunks = len(data)
    print(f"\n[System] Successfully loaded {total_chunks} chunks from JSON.")
    print("[System] Starting Vectorization and Database Injection Pipeline...\n")

    for i, item in enumerate(data):
        project_name = item.get("project_name", "General")
        section_name = item.get("section_name", "Unknown")
        chunk_content = item.get("chunk_content", "")

        print(f"Processing [{i+1}/{total_chunks}]: {project_name} -> {section_name}")

        try:
            # Step 1: Generate the vector embedding (The mathematical representation of the text)
            embedding = encoder.encode(chunk_content).tolist()

            # Step 2: Prepare the strict payload for Supabase
            payload = {
                "project_name": project_name,
                "section_name": section_name,
                "chunk_content": chunk_content,
                "embedding": embedding
            }

            # Step 3: Execute the database insert operation
            response = supabase.table("portfolio_chunks").insert(payload).execute()
            print(f"  --> Success: Embedded and stored in Supabase.")
            
        except Exception as e:
            print(f"  --> ERROR failed to inject chunk: {e}")

    print("\n[System] All data successfully injected into the vector database!")
    print("[System] Phase 1 Complete. You are ready to build the RAG Backend.")

if __name__ == "__main__":
    inject_data_to_db()