from openai import OpenAI 
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the OpenAI client
client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")

def chat_with_gpt4o(system_prompt, user_prompt):
    """Generates a response from GPT-4o."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
                # {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.8
        )

        # Access the response using dot notation
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content.strip()
        else:
            print("Invalid response format:", response)
            return "No valid response generated."

    except Exception as e:
        print(f"Unexpected Error: {e}")
        return f"Error: {str(e)}"

