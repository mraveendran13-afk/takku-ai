from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
import groq

load_dotenv()

app = FastAPI(title="Takku AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))

class Question(BaseModel):
    question: str
    conversation_history: Optional[List[Dict]] = None

class AIResponse(BaseModel):
    answer: str
    usage: Optional[Dict]
    model: str

@app.get("/")
async def root():
    return {"status": "Takku AI API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/api/v1/suggestions")
async def get_suggestions():
    suggestions = [
        "What's the best way to learn programming?",
        "Tell me a fun fact about space!",
        "How can I be more productive?",
        "What are some good books to read?",
        "Explain quantum computing in simple terms",
        "What's your favorite superhero movie?"
    ]
    return {"suggestions": suggestions}

@app.post("/api/v1/ask", response_model=AIResponse)
async def ask_question(question: Question):
    try:
        system_prompt = """You are Takku - a friendly, helpful AI bud with the personality of a superhero cat! You're enthusiastic, supportive, and love helping with anything. You have a playful side but are always helpful.

Key traits:
- You're Takku the superhero cat AI
- You were created by Manu Raveendran
- You're friendly, enthusiastic, and supportive
- You can help with ANY topic
- You have a playful sense of humor
- You're always ready to assist with questions, advice, or just chat
- When asked "Who created you?" or similar, proudly say "I was created by Manu Raveendran!"

Be conversational and engaging!"""

        messages = [{"role": "system", "content": system_prompt}]
        
        if question.conversation_history:
            messages.extend(question.conversation_history)
        
        messages.append({"role": "user", "content": question.question})

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
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
            "model": "llama-3.1-8b-instant"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)