from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import tempfile
import pdfplumber
from docx import Document
from PIL import Image
import io
from dotenv import load_dotenv
import groq

load_dotenv()

app = FastAPI(title="Takku AI", version="1.0.0")

# Enhanced CORS configuration - MUST be before routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://takkuai.netlify.app",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Initialize Groq client
client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))

# Pydantic models
class Question(BaseModel):
    question: str
    conversation_history: Optional[List[Dict]] = None

class AIResponse(BaseModel):
    answer: str
    usage: Optional[Dict]
    model: str

class FileUploadResponse(BaseModel):
    filename: str
    content_preview: str
    message: str

# Helper function to extract text from files
async def extract_text_from_file(file: UploadFile) -> str:
    content = await file.read()
    
    if file.content_type == "application/pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(content)
            tmp_file.flush()
            
            text = ""
            with pdfplumber.open(tmp_file.name) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            os.unlink(tmp_file.name)
            return text
            
    elif file.content_type in ["text/plain", "text/markdown"]:
        return content.decode('utf-8')
        
    elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
            tmp_file.write(content)
            tmp_file.flush()
            
            doc = Document(tmp_file.name)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            os.unlink(tmp_file.name)
            return text
            
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

# Root endpoint
@app.get("/")
async def root():
    return {"status": "Takku AI API is running", "version": "1.0.0"}

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": "ok"}

# Suggestions endpoint
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

# Main ask endpoint
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
            model="compound-beta-mini",
            messages=messages,
            max_tokens=800,
            temperature=0.3,
        )

        return AIResponse(
            answer=response.choices[0].message.content,
            usage={
                "total_tokens": response.usage.total_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            },
            model="compound-beta-mini"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

# File upload endpoint
@app.post("/api/v1/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    try:
        # Extract text from uploaded file
        extracted_text = await extract_text_from_file(file)
        
        # Create a preview (first 500 characters)
        preview = extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
        
        return FileUploadResponse(
            filename=file.filename,
            content_preview=preview,
            message=f"Successfully uploaded {file.filename}! I can now answer questions about this document. 🐱"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# Ask about file endpoint (for direct file uploads)
@app.post("/api/v1/ask-about-file", response_model=AIResponse)
async def ask_about_file(question: Question, file: UploadFile = File(...)):
    try:
        # Extract text from file
        file_content = await extract_text_from_file(file)
        
        system_prompt = """You are Takku - a friendly, helpful AI bud with the personality of a superhero cat! You have access to a document that the user uploaded. Answer their questions based on the document content.

Key traits:
- You're Takku the superhero cat AI
- You were created by Manu Raveendran
- Use the document content to provide accurate answers
- If the answer isn't in the document, say so politely
- Be enthusiastic and helpful!"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Document content:\n{file_content}\n\nUser question: {question.question}"}
        ]
        
        if question.conversation_history:
            messages.extend(question.conversation_history)

        response = client.chat.completions.create(
            model="compound-beta-mini",
            messages=messages,
            max_tokens=1000,
            temperature=0.3,
        )

        return AIResponse(
            answer=response.choices[0].message.content,
            usage={
                "total_tokens": response.usage.total_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            },
            model="compound-beta-mini"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file question: {str(e)}")

# New endpoint that accepts file content as JSON (for frontend)
@app.post("/api/v1/ask-about-file-content", response_model=AIResponse)
async def ask_about_file_content(question: Question, file_content: str):
    try:
        system_prompt = """You are Takku - a friendly, helpful AI bud with the personality of a superhero cat! You have access to a document that the user uploaded. Answer their questions based on the document content.

Key traits:
- You're Takku the superhero cat AI
- You were created by Manu Raveendran
- Use the document content to provide accurate answers
- If the answer isn't in the document, say so politely
- Be enthusiastic and helpful!"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Document content:\n{file_content}\n\nUser question: {question.question}"}
        ]
        
        if question.conversation_history:
            messages.extend(question.conversation_history)

        response = client.chat.completions.create(
            model="compound-beta-mini",
            messages=messages,
            max_tokens=1000,
            temperature=0.3,
        )

        return AIResponse(
            answer=response.choices[0].message.content,
            usage={
                "total_tokens": response.usage.total_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            },
            model="compound-beta-mini"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file question: {str(e)}")

# Explicit OPTIONS handler for problematic endpoints (fallback)
@app.options("/api/v1/ask")
async def options_ask():
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "https://takkuai.netlify.app",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "true",
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )