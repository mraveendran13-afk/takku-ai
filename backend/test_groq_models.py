from groq import Groq
import os

def check_groq_models():
    try:
        client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
        
        print("Checking available Groq models...")
        models = client.models.list()
        
        print("\nAvailable Groq models:")
        for model in models.data:
            print(f"- {model.id}")
            
        # Check for embedding models
        print("\nLooking for embedding models...")
        embedding_models = [model.id for model in models.data if 'embed' in model.id.lower()]
        
        if embedding_models:
            print("Found embedding models:")
            for model in embedding_models:
                print(f"- {model}")
        else:
            print("No embedding models found in Groq API")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_groq_models()