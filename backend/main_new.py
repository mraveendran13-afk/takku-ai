from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
import uuid
from datetime import datetime
from typing import Optional

# Pinecone import
try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    print("❌ Pinecone not installed. Running without memory features.")
    PINECONE_AVAILABLE = False

# Sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("❌ Sentence transformers not available. Using simple embeddings.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Initialize FastAPI
app = FastAPI(title="Takku Medical AI", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client with fallback
groq_api_key = os.environ.get("GROQ_API_KEY")
if groq_api_key and groq_api_key != "your_actual_groq_api_key_here":
    client = Groq(api_key=groq_api_key)
    GROQ_AVAILABLE = True
    print("✅ Groq client initialized successfully!")
else:
    client = None
    GROQ_AVAILABLE = False
    print("⚠️  Groq API key not found - running in demo mode")

# Initialize Pinecone
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME")
pinecone_environment = os.environ.get("PINECONE_ENVIRONMENT")

# Initialize embedding model
if SENTENCE_TRANSFORMERS_AVAILABLE:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
else:
    embedding_model = None

# Initialize Pinecone
index = None
if PINECONE_AVAILABLE and pinecone_api_key and pinecone_environment:
    try:
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
        index = pinecone.Index(pinecone_index_name)
        print("✅ Pinecone connected successfully!")
    except Exception as e:
        print(f"❌ Pinecone connection failed: {e}")
        index = None
else:
    print("⚠️  Pinecone not configured - running without memory")

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    symptoms: Optional[str] = ""

class SearchRequest(BaseModel):
    query: str

def get_embedding(text):
    """Generate embedding for text"""
    if embedding_model:
        return embedding_model.encode(text).tolist()
    else:
        # Simple fallback
        import random
        return [random.random() for _ in range(384)]

def store_conversation_memory(user_id: str, user_message: str, assistant_response: str, conversation_context: str):
    """Store conversation in Pinecone memory"""
    if index is None:
        return
        
    conversation_text = f"User: {user_message}\nAssistant: {assistant_response}\nContext: {conversation_context}"
    embedding = get_embedding(conversation_text)
    
    metadata = {
        "user_id": user_id,
        "user_message": user_message,
        "assistant_response": assistant_response,
        "conversation_context": conversation_context,
        "timestamp": datetime.utcnow().isoformat(),
        "type": "conversation_memory"
    }
    
    memory_id = str(uuid.uuid4())
    index.upsert(vectors=[(memory_id, embedding, metadata)])
    print(f"💾 Memory stored: {memory_id}")

def search_related_memories(user_id: str, current_query: str, top_k: int = 3):
    """Search for related past conversations"""
    if index is None:
        return []
    
    query_embedding = get_embedding(current_query)
    
    try:
        results = index.query(
            vector=query_embedding,
            filter={"user_id": user_id, "type": "conversation_memory"},
            top_k=top_k,
            include_metadata=True
        )
        
        relevant_memories = []
        for match in results.matches:
            if match.score > 0.7:
                memory_data = match.metadata
                relevant_memories.append({
                    "user_message": memory_data.get("user_message", ""),
                    "assistant_response": memory_data.get("assistant_response", ""),
                    "context": memory_data.get("conversation_context", ""),
                    "similarity_score": match.score,
                    "timestamp": memory_data.get("timestamp", "")
                })
        
        print(f"🔍 Found {len(relevant_memories)} relevant memories")
        return relevant_memories
        
    except Exception as e:
        print(f"❌ Memory search error: {e}")
        return []

def build_context_from_memories(user_id: str, current_query: str, current_symptoms: str):
    """Build context from relevant past conversations"""
    relevant_memories = search_related_memories(user_id, current_query)
    
    if not relevant_memories:
        return current_symptoms
    
    context = f"Current symptoms: {current_symptoms}\n\n"
    context += "Relevant past conversations:\n"
    
    for i, memory in enumerate(relevant_memories, 1):
        context += f"{i}. Previous concern: {memory['user_message']}\n"
        context += f"   My advice: {memory['assistant_response']}\n"
        context += f"   Context: {memory['context']}\n\n"
    
    return context

def get_user_id_from_request():
    """Generate user ID (simplified for FastAPI)"""
    return str(uuid.uuid4())  # In production, use proper user authentication

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        user_message = request.message.strip()
        symptoms = request.symptoms.strip()
        
        if not user_message:
            raise HTTPException(status_code=400, detail="No message provided")
        
        user_id = get_user_id_from_request()
        print(f"👤 User {user_id}: {user_message}")
        
        # Build enhanced context with memory
        enhanced_context = build_context_from_memories(user_id, user_message, symptoms)
        
        # Create conversation prompt
        system_prompt = f"""You are Takku, a compassionate medical AI assistant. 

Current Context:
{enhanced_context}

User's current message: {user_message}

Guidelines:
- Provide empathetic, professional medical guidance
- Reference relevant past conversations when helpful
- Maintain continuity in ongoing health discussions
- Always suggest professional healthcare for serious concerns
- Be clear about your limitations as an AI
- Focus on education and support, not diagnosis"""
        
        # Get response from Groq or use demo response
        if client:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                model="llama3-8b-8192",
                temperature=0.7,
                max_tokens=1024
            )
            assistant_response = chat_completion.choices[0].message.content
        else:
            # Demo response when Groq is not available
            assistant_response = f"I understand you're asking about: '{user_message}'. In a full setup, I would provide detailed medical guidance based on your symptoms: {symptoms}. The memory system is {'active' if index else 'inactive'}."
        
        # Store conversation in memory
        store_conversation_memory(
            user_id=user_id,
            user_message=user_message,
            assistant_response=assistant_response,
            conversation_context=enhanced_context
        )
        
        print(f"🤖 Takku: {assistant_response[:100]}...")
        
        return {
            'response': assistant_response,
            'memory_used': len(enhanced_context) > len(symptoms),
            'memory_system': 'active' if index else 'inactive',
            'groq_available': GROQ_AVAILABLE
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    return {
        'status': 'healthy', 
        'memory_system': 'active' if index else 'inactive',
        'pinecone_available': PINECONE_AVAILABLE,
        'groq_available': GROQ_AVAILABLE,
        'service': 'Takku Medical AI'
    }

@app.post("/memories/search")
async def search_memories(request: SearchRequest):
    try:
        user_id = get_user_id_from_request()
        memories = search_related_memories(user_id, request.query)
        
        return {
            'query': request.query,
            'memories_found': len(memories),
            'memories': memories
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)