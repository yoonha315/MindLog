# ──────────────────────────────────────────────────────────────────
# MindLog — LLM Extraction Benchmark
# ──────────────────────────────────────────────────────────────────
# Reads synthetic conversations, sends each to OpenAI API,
# extracts 5 structured fields, and saves results to JSON.
#
# Usage: python evaluation/extract_benchmark.py
# Output: evaluation/extraction_results.json
# ──────────────────────────────────────────────────────────────────

import json
import time
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

# ── Load environment variables ──────────────────────────────────
load_dotenv()
client = OpenAI()

# ── Model config (change this to try different models) ──────────
MODEL = "gpt-4o"

# ── File paths ──────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(SCRIPT_DIR, "synthetic_conversations.json")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "extraction_results.json")

# ── Extraction prompt ──────────────────────────────────────────
SYSTEM_PROMPT = """You are a clinical NLP extraction module for a mental health self-monitoring app.

Given a user's daily conversational log (in Korean), extract these 5 fields.
Choose EXACTLY one value from the allowed options for each field.

1. affect: positive / neutral / negative
2. energy: low / medium / high
3. sleep_quality: good / fair / poor / not_mentioned
4. medication_taken: yes / no / not_mentioned
5. dominant_theme: work / relationships / physical_health / finances / daily_life / emotional

Rules:
- If the user does NOT mention sleep at all, choose "not_mentioned".
- If the user does NOT mention medication at all, choose "not_mentioned".
- For dominant_theme, pick the SINGLE most prominent topic.
- Respond with ONLY valid JSON. No explanation, no markdown, no backticks."""


# ── Main extraction loop ────────────────────────────────────────
def extract_fields(conversation_text):
    """Send one conversation to the API and parse the extracted fields."""
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,  # Deterministic output for reproducibility
        max_tokens=200,  # Structured JSON response is short
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": conversation_text}
        ]
    )

    # Parse the JSON response
    raw = response.choices[0].message.content.strip()

    # Handle cases where model wraps JSON in backticks
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    return json.loads(raw)


def main():
    # Load synthetic conversations
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    print(f"Loaded {len(conversations)} conversations from {INPUT_PATH}")
    print(f"Using model: {MODEL}")
    print(f"{'─' * 50}")

    results = []

    for item in conversations:
        conv_id = item["id"]
        print(f"  Extracting conversation {conv_id}/{len(conversations)}...", end=" ")

        try:
            extracted = extract_fields(item["conversation"])
            results.append({
                "id": conv_id,
                "extracted": extracted
            })
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "id": conv_id,
                "extracted": None,
                "error": str(e)
            })

        # Wait 1 second between calls to avoid rate limits
        time.sleep(1)

    # Save results
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"{'─' * 50}")
    print(f"Done! Results saved to {OUTPUT_PATH}")
    print(f"Successfully extracted: {sum(1 for r in results if r['extracted'])} / {len(conversations)}")


if __name__ == "__main__":
    main()