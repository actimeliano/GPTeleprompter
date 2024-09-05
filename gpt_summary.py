import openai
import os

# Set up OpenAI API key
openai.api_key = os.environ.get('OPENAI_API_KEY')

if not openai.api_key:
    print("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    exit(1)

def get_gpt_summary(full_text, current_summary):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes spoken content."},
            {"role": "user", "content": f"Here's the current summary:\n{current_summary}\n\nHere's the full transcription so far:\n{full_text}\n\nPlease provide an updated summary of the content, focusing on the main points and any new information."}
        ]
        
        response = openai.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",  # Use an appropriate model
            messages=messages,
            max_tokens=150  # Adjust as needed
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred while generating the summary: {str(e)}"