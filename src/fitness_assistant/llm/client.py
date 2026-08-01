
import os
import dotenv
from groq import Groq

dotenv.load_dotenv()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)
def llm(prompt, model="llama-3.3-70b-versatile"):   
    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    model=model,
    )
    return chat_completion.choices[0].message.content