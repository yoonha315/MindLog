"""OpenAI client initialization."""

import os

from openai import OpenAI


def build_client() -> OpenAI:
    """
    Initialize the OpenAI client using the OPENAI_API_KEY environment variable.

    Raises ValueError if OPENAI_API_KEY is not set.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment. "
            "Create a .env file in the project root with: OPENAI_API_KEY=sk-..."
        )
    return OpenAI(api_key=api_key)
