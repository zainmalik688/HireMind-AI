import os
from pathlib import Path
from google import genai
from fastapi import HTTPException
from dotenv import load_dotenv

# Use the current working directory where the server is running
root_dir = Path.cwd()
env_path = root_dir / ".env"

# Explicitly load the file from the absolute path
load_dotenv(dotenv_path=env_path)

class AIService:
    @staticmethod
    def analyze_resume(extracted_text: str) -> str:
        # Retrieve the generic API key from environment
        api_key = os.environ.get("API_KEY")
        
        if not api_key:
            raise HTTPException(
                status_code=500, 
                detail=f"Model API key configuration is missing on the server. Looked at path: {env_path}"
            )
            
        try:
            # Initialize the client with the key
            client = genai.Client(api_key=api_key)
            
            prompt = (
                "You are an expert ATS optimization engine. Analyze the following resume text. "
                "Provide a structured breakdown of key skills, professional formatting critique, "
                "and concrete recommendations for alignment with production engineering roles:\n\n"
                f"{extracted_text}"
            )
            
            # Call the high-performance model flash variant
            response = client.models.generate_content(
                model='gemini-3.5-flash',
                contents=prompt,
            
            )
            
            return response.text
            
        except Exception as e:
            raise HTTPException(
                status_code=502, 
                detail=f"Inference engine failure: {str(e)}"
            )