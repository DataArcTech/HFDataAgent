import os
from openai import AsyncOpenAI

async def chat_complete(prompt, model="deepseek-chat", temperature=0):
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    
    response = await client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    return response.choices[0].message.content