from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Medical AI Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get("/api/v1/suggestions")
async def get_suggestions():
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
    try:
        # Mock medical responses
        mock_responses = {
            "hypertension": """**Hypertension (High Blood Pressure)**

**Overview:**
Hypertension is a condition where blood pressure in arteries is persistently elevated.

**Symptoms:**
- Often no symptoms (silent condition)
- Severe cases: headaches, shortness of breath, nosebleeds
- Advanced stages: vision changes, chest pain

**Causes:**
- Primary: Genetics, age, lifestyle factors
- Secondary: Kidney disease, thyroid issues, medications
- Risk factors: High salt diet, obesity, stress, smoking

**Diagnosis:**
- Blood pressure readings >130/80 mmHg
- Multiple measurements over time
- Ambulatory monitoring for accuracy

**Treatment:**
- Lifestyle changes: DASH diet, regular exercise, weight management
- Medications: ACE inhibitors, beta-blockers, diuretics
- Regular monitoring and follow-up care

**When to Seek Help:**
- BP readings consistently above 180/120 mmHg
- Severe headache, chest pain, or vision changes
- Symptoms of stroke or heart attack""",

            "diabetes": """**Diabetes Management**

**Overview:**
Diabetes affects how your body processes blood sugar (glucose).

**Symptoms:**
- Increased thirst and frequent urination
- Extreme fatigue and blurred vision
- Slow-healing wounds
- Unexplained weight loss

**Diagnosis:**
- Fasting blood sugar tests
- A1C hemoglobin tests
- Oral glucose tolerance tests

**Management:**
- Blood sugar monitoring
- Insulin therapy (Type 1 diabetes)
- Oral medications (Type 2 diabetes)
- Carbohydrate counting and meal planning
- Regular physical activity

**Important:**
Always consult healthcare professionals for personalized diabetes management.""",

            "default": """Thank you for your medical question about "{}".

This is a mock response from the Medical AI Assistant. For comprehensive, real-time medical information, please ensure your OpenAI API key has sufficient quota or set up billing.

The system is working correctly - the frontend and backend are communicating properly. To get real AI-powered medical answers, please set up your OpenAI billing at: https://platform.openai.com/account/billing/overview

In a production environment, this would provide detailed medical information about your query.""".format(medical_question.question)
        }

        question_lower = medical_question.question.lower()
        
        if "hypertension" in question_lower or "blood pressure" in question_lower:
            answer = mock_responses["hypertension"]
        elif "diabetes" in question_lower:
            answer = mock_responses["diabetes"]
        else:
            answer = mock_responses["default"]

        return {
            "answer": answer,
            "usage": {"total_tokens": 150, "prompt_tokens": 50, "completion_tokens": 100},
            "model": "mock-medical-assistant"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)