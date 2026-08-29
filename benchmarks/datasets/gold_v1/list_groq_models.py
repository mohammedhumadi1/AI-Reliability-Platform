"""List all models actually available to this Groq account."""
from __future__ import annotations

import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

models = client.models.list()

print("Available models for this account:\n")
for model in models.data:
    print(f"- {model.id}")
