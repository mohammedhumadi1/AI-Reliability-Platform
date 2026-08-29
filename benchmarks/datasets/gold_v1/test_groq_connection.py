"""
Quick sanity check that GROQ_API_KEY is loaded correctly and the
Groq API responds. This is NOT part of the benchmark dataset build —
just a connectivity/credentials test before running the real 24 questions.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise SystemExit(
        "GROQ_API_KEY not found. Check that it is set in your .env file."
    )

client = Groq(api_key=api_key)

response = client.chat.completions.create(
    model="qwen/qwen3.6-27b",
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly one word: 'connected'.",
        }
    ],
    max_tokens=10,
)

print("Model used:", response.model)
print("Response:", response.choices[0].message.content)
print("\nGROQ_API_KEY is working correctly.")
