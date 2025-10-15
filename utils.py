from openai import OpenAI
from config import LLM_API_KEY, LLM_BASE_URL

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

def chat_complete(prompt):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    return response.choices[0].message.content