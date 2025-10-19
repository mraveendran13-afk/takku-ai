from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
import groq

# Load environment variables
load_dotenv()

app = FastAPI(title="Medical AI Assistant", version="1.0.0")

# CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Groq client
client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))

class MedicalQuestion(BaseModel):
    question: str
    conversation_history: Optional[List[Dict]] = None

class MedicalResponse(BaseModel):
    answer: str
    usage: Optional[Dict]
    model: str

@app.get("/")
async def root():
    return {"status": "Medical AI Assistant API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/api/v1/suggestions")
async def get_suggestions():
    """Get common medical question suggestions"""
    suggestions = [
        "What are the symptoms and treatments for hypertension?",
        "How is diabetes diagnosed and managed?",
        "What are the warning signs of a heart attack?",
        "Describe appendicitis symptoms and emergency care",
        "What causes migraines and how are they treated?",
        "Explain asthma symptoms and management strategies"
    ]
    return {"suggestions": suggestions}

@app.post("/api/v1/ask", response_model=MedicalResponse)
async def ask_medical_question(medical_question: MedicalQuestion):
    """Endpoint for medical questions"""
    try:
        # Enhanced medical prompt
        system_prompt = """You are a medical AI assistant. Provide comprehensive, accurate medical information based on established medical knowledge.

Always structure your answers to include:
1. **Overview**: Clear definition and explanation
2. **Symptoms**: Key signs and symptoms
3. **Causes**: Underlying causes and risk factors  
4. **Diagnosis**: How it's typically diagnosed
5. **Treatment**: Available treatments and management
6. **When to Seek Help**: Important warning signs

Be thorough but concise. Always emphasize consulting healthcare professionals for medical advice."""

        # Build conversation context
        messages = [{"role": "system", "content": system_prompt}]
        
        if medical_question.conversation_history:
            messages.extend(medical_question.conversation_history)
        
        messages.append({"role": "user", "content": medical_question.question})

        # Call Groq API - Using Llama 3.1 70B model (free tier)
        response = client.chat.completions.create(
            model="llama-3.1-70b-versatile",  # Free model on Groq
            messages=messages,
            max_tokens=800,
            temperature=0.3,
        )

        return {
            "answer": response.choices[0].message.content,
            "usage": {
                "total_tokens": response.usage.total_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            },
            "model": "llama-3.1-70b-versatile"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)