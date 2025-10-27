from openai import AsyncOpenAI
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

async def chat_complete(prompt, temperature=0):
    response = await client.chat.completions.create(
        model=LLM_MODEL,
        temperature=temperature,
        messages=[
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    return response.choices[0].message.content