import os
import requests
import json
from dotenv import load_dotenv
from fastapi import HTTPException

# Load environment variables
load_dotenv()

API_KEY = os.getenv("GENAI_API_KEY")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Function to generate text using Gemini API
def generate_text(prompt: str) -> str:
    """Generate text using Gemini API."""
    url = f"{BASE_URL}?key={API_KEY}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    try:
        # Send request to Gemini API
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()

        # Parse the response
        result = response.json()
        gemini_response = (
            result.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "No response")
        )

        return gemini_response

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
