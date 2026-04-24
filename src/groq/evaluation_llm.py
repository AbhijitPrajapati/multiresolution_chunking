import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def chat_response(prompt):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="openai/gpt-oss-120b",
    )

    return chat_completion.choices[0].message.content
