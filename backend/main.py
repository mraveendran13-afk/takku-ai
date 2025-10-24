from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
import uuid
from datetime import datetime
from typing import Optional, List, Dict

# Pinecone import
try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    print("❌ Pinecone not installed. Running without memory features.")
    PINECONE_AVAILABLE = False

# Initialize FastAPI
app = FastAPI(title="Takku AI", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client
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
    use_web_search: Optional[bool] = True

class SearchRequest(BaseModel):
    query: str

class Question(BaseModel):
    question: str
    conversation_history: Optional[List[Dict]] = None
    game_context: Optional[str] = None

class AIResponse(BaseModel):
    answer: str
    usage: Optional[Dict]
    model: str
    searched_web: Optional[bool] = False
    search_query: Optional[str] = None

def get_embedding(text):
    """Generate simple deterministic embedding for text"""
    import hashlib
    import numpy as np
    
    hash_obj = hashlib.md5(text.encode())
    hash_int = int(hash_obj.hexdigest()[:8], 16)
    
    np.random.seed(hash_int)
    return np.random.rand(384).tolist()

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
    return str(uuid.uuid4())

def needs_web_search(message: str) -> bool:
    """Detect if message needs current information"""
    keywords = [
        'today', 'latest', 'current', 'news', 'recent', 'now',
        'this week', 'this month', 'this year', '2024', '2025',
        'weather', 'trending', 'happening', 'update', 'breaking'
    ]
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in keywords)

@app.post("/chat")
async def chat(request: ChatRequest):
    """Original chat endpoint with memory"""
    try:
        user_message = request.message.strip()
        symptoms = request.symptoms.strip()
        
        if not user_message:
            raise HTTPException(status_code=400, detail="No message provided")
        
        user_id = get_user_id_from_request()
        print(f"👤 User {user_id}: {user_message}")
        
        # Build enhanced context with memory
        enhanced_context = build_context_from_memories(user_id, user_message, symptoms)
        
        # For now, always use regular model (compound models need special handling)
        model = "llama3-8b-8192"
        use_compound = False  # Disabled until we fix compound model support
        
        print(f"🤖 Using model: {model}")
        
        # Create conversation prompt
        system_prompt = f"""You are Takku, a compassionate and friendly superhero cat AI assistant! 

Current Context:
{enhanced_context}

User's current message: {user_message}

Guidelines:
- Be empathetic, friendly, and helpful
- Reference relevant past conversations when helpful
- Maintain continuity in discussions
- Always suggest professional help for serious medical concerns
- Be clear about your limitations as an AI
- Focus on education and support
- Current date: {datetime.now().strftime('%B %d, %Y')}"""
        
        # Get response from Groq
        if client:
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    model=model,
                    temperature=0.7,
                    max_tokens=1024
                )
                
                assistant_response = chat_completion.choices[0].message.content
                searched_web = False  # Web search disabled for now
                
            except Exception as e:
                print(f"⚠️ Groq API error: {e}")
                assistant_response = f"I'm having trouble accessing my full capabilities right now. However, regarding '{user_message}', I can tell you that the memory system is {'active' if index else 'inactive'}."
                searched_web = False
        else:
            assistant_response = f"I understand you're asking about: '{user_message}'. In a full setup, I would provide detailed guidance. The memory system is {'active' if index else 'inactive'}."
            searched_web = False
        
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
            'groq_available': GROQ_AVAILABLE,
            'searched_web': searched_web,
            'model_used': model
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/v1/ask", response_model=AIResponse)
async def ask_question(question: Question):
    """New endpoint for general Q&A"""
    try:
        # Use game context if provided, otherwise use default
        if question.game_context:
            system_prompt = question.game_context
            model = "llama-3.1-8b-instant"
        else:
            # For now, always use regular model (compound models need special handling)
            model = "llama-3.1-8b-instant"
            
            system_prompt = f"""You are Takku - a friendly, helpful AI bud with the personality of a superhero cat! You're enthusiastic, supportive, and love helping with anything.

Key traits:
- You're Takku the superhero cat AI
- You were created by Manu Raveendran
- You're friendly, enthusiastic, and supportive
- You can help with ANY topic
- You have a playful sense of humor
- You're always ready to assist with questions, advice, or just chat
- When asked "Who created you?" or similar, proudly say "I was created by Manu Raveendran!"
- Current date: {datetime.now().strftime('%B %d, %Y')}

Be conversational and engaging!"""

        print(f"🤖 Using model: {model}")

        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        if question.conversation_history:
            messages.extend(question.conversation_history)
        
        # Add user question
        messages.append({"role": "user", "content": question.question})

        # Create completion
        if not client:
            raise HTTPException(status_code=503, detail="Groq API not available")
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=800,
                temperature=0.3,
            )

            searched_web = False  # Web search disabled for now
            
            print(f"✅ Response generated.")

            return AIResponse(
                answer=response.choices[0].message.content,
                usage={
                    "total_tokens": response.usage.total_tokens,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                },
                model=model,
                searched_web=searched_web,
                search_query=question.question if searched_web else None
            )
        
        except Exception as e:
            print(f"Error in Groq API call: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in ask_question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@app.get("/")
async def root():
    return {
        "status": "Takku AI is running",
        "version": "2.0.0",
        "features": ["chat", "memory"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {
        'status': 'healthy', 
        'memory_system': 'active' if index else 'inactive',
        'pinecone_available': PINECONE_AVAILABLE,
        'groq_available': GROQ_AVAILABLE,
        'web_search': 'disabled',  # Changed to disabled
        'service': 'Takku AI'
    }

@app.get("/api/v1/suggestions")
async def get_suggestions():
    """Get conversation suggestions"""
    suggestions = [
        "What's the best way to learn programming?",
        "Tell me a fun fact about space!",
        "What are some good books to read?",
        "Tell me about the latest AI developments"
    ]
    return {"suggestions": suggestions}

@app.post("/memories/search")
async def search_memories(request: SearchRequest):
    """Search conversation memories"""
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