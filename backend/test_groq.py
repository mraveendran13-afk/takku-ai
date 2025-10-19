import os
from dotenv import load_dotenv
import groq

load_dotenv()

print("Testing Groq connection...")
api_key = os.getenv("GROQ_API_KEY")
print(f"API Key exists: {bool(api_key)}")
print(f"API Key starts with: {api_key[:10] if api_key else 'None'}...")

try:
    client = groq.Client(api_key=api_key)
    print("✅ Groq client created successfully")
    
    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[{"role": "user", "content": "Say just 'Hello' in one word"}],
        max_tokens=5
    )
    print("✅ Groq API call successful!")
    print(f"Response: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"❌ ERROR: {str(e)}")