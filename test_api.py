# ──────────────────────────────────────────
# MindLog — API Connection Test
# ──────────────────────────────────────────
# Run this after setup to verify your API connection.
# If successful, you'll see a response from GPT.
# Safe to delete after confirming it works.
#
# Usage: python test_api.py
# ──────────────────────────────────────────

from dotenv import load_dotenv  # Load env vars from .env file
from openai import OpenAI       # OpenAI API client

# Register OPENAI_API_KEY from .env as an environment variable
load_dotenv()

# Initialize client (auto-reads API key from environment)
client = OpenAI()

# Send a test message to GPT
response = client.chat.completions.create(
    model="gpt-4o",              # Using GPT-4o
    max_tokens=100,              # Max response length
    messages=[
        {"role": "user", "content": "Say hello in one sentence."}
    ]
)

# Print response — if you see a sentence here, setup is complete!
print(response.choices[0].message.content)