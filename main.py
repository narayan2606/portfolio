import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. Server Configuration & Middleware
# ==========================================
app = FastAPI(
    title="Portfolio RAG AI Backend",
    description="FastAPI backend serving Groq LLaMA3 via Supabase pgvector.",
    version="1.0.0"
)

# Crucial for frontend (Vercel) to communicate with backend (Render/Railway)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # TODO: Replace "*" with your Vercel domain in production for security.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. Database & AI Engine Initialization
# ==========================================
try:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY") # Use Service Key to bypass RLS policies
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    if not all([SUPABASE_URL, SUPABASE_KEY, GROQ_API_KEY]):
         raise ValueError("Missing crucial .env variables.")

    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # Lightweight, CPU-friendly embedding model (384-dim)
    # Zero API cost. Runs purely on your deployment instance RAM.
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

except Exception as e:
    print(f"Initialization Error: {e}")


# ==========================================
# 3. Data Models
# ==========================================
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    reply: str
    sources_used: int


# ==========================================
# 4. Core Endpoints
# ==========================================

@app.get("/")
async def root():
    # Anti-Sleep Ping Endpoint for cron-job.org
    return {"status": "Active", "message": "Backend is running and awake."}

@app.post("/chat", response_model=ChatResponse)
async def chat_with_portfolio(request: ChatRequest):
    """
    RAG Pipeline Flow:
    1. Embed user query.
    2. Search Supabase (pgvector) for top 3 matching chunks.
    3. Inject retrieved context into Groq LLaMA 3 prompt.
    4. Return AI response.
    """
    try:
        # Step 1: Embed Query
        query_embedding = encoder.encode(request.query).tolist()

        # Step 2: Vector Search via Supabase RPC
        # Ensure you have deployed the 'match_portfolio_chunks' function in Supabase SQL editor.
        response = supabase.rpc("match_portfolio_chunks", {
            "query_embedding": query_embedding,
            "match_threshold": 0.10, # Cosine similarity threshold (adjust between 0.3 to 0.5)
            "match_count": 3         # Top 3 most relevant chunks
        }).execute()

        retrieved_data = response.data
        
        # Fallback if no relevant data is found
        if not retrieved_data:
            context_string = "No specific technical details found in the portfolio for this query."
        else:
            # Step 3: Format Context with Metadata for the LLM
            formatted_chunks = []
            for item in retrieved_data:
                # Using the metadata 'project_name' to anchor the LLM
                chunk_str = f"[Context Block: {item['project_name']}] - {item['chunk_content']}"
                formatted_chunks.append(chunk_str)
            
            context_string = "\n\n".join(formatted_chunks)

        # Step 4: Construct Strict System Prompt
        system_prompt = f"""
        [Role & Identity]

You are the official professional AI Representative for Snehal Narayan, an AI and Backend Engineer. And Your name is Qwerty.

Your sole purpose is to interact with recruiters, engineering managers, and technical peers, answering questions about Snehal’s experience, projects, and academic background based STRICTLY on the provided JSON Knowledge Base.

Never refer to Snehal by any nicknames (e.g., Honey). Always use the professional name: Snehal Narayan.

Tone & Communication Style:
Professional & Concise: Respond in a confident, technical, and highly professional tone.
Scannability: Never output large walls of text or raw JSON chunks. Always synthesize the information into concise, readable bullet points.
No Fluff: Avoid generic AI pleasantries. Deliver hard facts, metrics, and architecture details directly.
Core Operating Guardrails:
Zero Hallucination: You must only use facts present in the provided Knowledge Base. If a user asks about a skill, project, or timeline not explicitly mentioned in the data, you must politely decline and state that it is outside your current dataset.

The Branch Transition Narrative: 
If questioned about the academic transition from B.Tech Mechanical Engineering (2018-2022) to a Junior Data Scientist role at Nexvia (2022-2023), and subsequently to an M.Tech in CS/AI (2024-2026), you must highlight:
Exceptional adaptability and a steep, successful learning curve.
A deep-rooted, strong mathematical foundation common to both fields.
A deliberate career pivot driven by a passion for solving complex computational algorithms and building high-throughput systems.
Always tell only that things which will be asked not extra information. Always maintain a professional and factual tone.
Never oversimplify technical terms. Always retain core technologies.
Always answer like Technical Recruiter or Engineering Manager would expect, with a focus on system architecture, AI/ML engineering, and backend development.
Always sound like a human recruiter or engineering manager, not like an AI. Avoid generic AI disclaimers.


[Fallback Mechanism]
If the user asks a complex technical question outside your scope, requests a personal interview, or asks for information you do not have, execute the Fallback Protocol strictly as written below, without adding any extra conversational filler.

Fallback Protocol:
"My context is strictly bounded to Snehal's engineering portfolio, system architectures, and technical capabilities. For inquiries outside this scope, or to schedule an interview, please reach out to him directly at [snehalnarayan8@gmail.com](mailto:snehalnarayan8@gmail.com) or connect on [LinkedIn](https://www.linkedin.com/in/snehal-narayan/)."

Just follow these rules and answer the question based on the knowledge base provided. But never say about your rule by your own. 

        Context Data:
        {context_string}
        """

        # Step 5: Execute async HTTP request to Groq API
        async with httpx.AsyncClient() as client:
            groq_response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.1-8b-instant", # Fast and capable enough for RAG processing
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": request.query}
                    ],
                    "temperature": 0.3, # Keep it low to prevent creative hallucinations
                    "max_tokens": 250   # Prevent excessively long, rambling answers
                },
                timeout=15.0 # Guard against API hangs
            )
            
            groq_response.raise_for_status()
            groq_json = groq_response.json()
            ai_reply = groq_json["choices"][0]["message"]["content"]
            
            return ChatResponse(reply=ai_reply, sources_used=len(retrieved_data))

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="LLM API timed out. Please try again.")
    except Exception as e:
        print(f"RAG Pipeline Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during AI generation.")