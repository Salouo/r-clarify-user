from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(".env")

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "user", "content": "Reply with exactly: hello"},
    ],
)

print(response.choices[0].message.content)
