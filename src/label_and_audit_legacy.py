#!/usr/bin/env python3
"""
Async batched labeling pipeline for pre-truncated conversation sequences.
- Labels each sequence using Groq (sync calls wrapped in asyncio.to_thread)
- Audits a random subset with OpenAI GPT-4o
- Logs performance, error rates, raw API responses, and audit disagreement
"""

import os
import json
import random
import asyncio
import time
import statistics
import re
from tqdm import tqdm
from dotenv import load_dotenv
from groq import Groq
from openai import AsyncOpenAI

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
load_dotenv()

INPUT_FILE = "data/cleaned_sequences.json"
GROQ_OUTPUT = "data/groq_labelled.json"
FINAL_OUTPUT = "data/verified_labelled.json"

BATCH_SIZE = int(os.getenv("LABEL_BATCH_SIZE", 8))
VERIFY_FRACTION = 0.1
RANDOM_SEED = 42

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------------------------------
# Intent taxonomy
# -------------------------------------------------------
INTENT_CATEGORIES_LIST_SIMPLIFIED = """→ academic_help – Students getting help with homework, assignments, tests...
→ writing_and_editing – General writing, essays, or educational text...
→ computer_programming – Code, debugging, or logic explanations...
→ how_to_advice – Step-by-step guidance or instructions...
→ creative_ideation – Brainstorming, creative ideas...
→ translation – Translate text...
→ greetings_and_chitchat – Small talk or casual chat...
→ specific_info – General factual queries not academic...
→ relationships_and_personal_reflection – Emotional or interpersonal topics...
→ games_and_role_play – Imaginative or character-based play...
→ media_generation_or_analysis – Work with media, art, or creative assets...
→ unclear – If there is no identifiable intent...
→ other – Anything else.
"""

LABEL_PROMPT = f"""
You are an intent classifier.
Classify the *user intent* of the following conversation into ONE category:

{INTENT_CATEGORIES_LIST_SIMPLIFIED}

Return ONLY JSON: {{"intent": "<category>"}}
"""

VERIFY_PROMPT = f"""
You are verifying an automatically generated intent classification.

If correct, respond:
{{"verified": true, "corrected_intent": "same"}}

If incorrect, respond:
{{"verified": false, "corrected_intent": "<new_intent>"}}

Available categories:
{INTENT_CATEGORIES_LIST_SIMPLIFIED}
"""

# -------------------------------------------------------
# Async helpers
# -------------------------------------------------------
async def groq_label_one(item, max_retries=3):
    text = item.get("sequence", "")
    for attempt in range(max_retries):
        try:
            # Run Groq call in a thread for async compatibility
            response = await asyncio.to_thread(
                groq_client.chat.completions.create,
                model="openai/gpt-4o-mini",
                messages=[
                    {"role": "system", "content": LABEL_PROMPT},
                    {"role": "user", "content": text}
                ],
                temperature=0
            )

            # --- Step 1: log raw response ---
            raw_response = response.choices[0].message.content.strip()
            item["groq_raw_response"] = raw_response[:500]  # store first 500 chars for logging
            # print(f"Raw Groq response: {raw_response[:200]}...")

            # --- Step 2: robust JSON parsing ---
            try:
                label_json = json.loads(raw_response)
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", raw_response, re.DOTALL)
                if match:
                    label_json = json.loads(match.group())
                else:
                    label_json = {"intent": "unclear"}

            item["intent"] = label_json.get("intent", "unclear")
            item["groq_status"] = "ok"
            return item

        except Exception as e:
            print(f"⚠️ Groq attempt {attempt+1}/{max_retries} failed: {e}\nContent snippet: {text[:50]}")
            item["groq_status"] = f"error: {e}"
            await asyncio.sleep(1)  # simple backoff

    item["intent"] = "unclear"
    return item


async def label_batch_groq(batch):
    tasks = [groq_label_one(item) for item in batch]
    return await asyncio.gather(*tasks)


async def verify_one_openai(msg):
    text = msg.get("sequence", "")
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": VERIFY_PROMPT},
                {"role": "user", "content": json.dumps({"content": text, "intent": msg["intent"]})}
            ],
            temperature=0
        )
        verification = json.loads(response.choices[0].message.content)
        msg["verified"] = verification.get("verified", False)
        msg["corrected_intent"] = (
            msg["intent"]
            if verification.get("corrected_intent") == "same"
            else verification.get("corrected_intent", msg["intent"])
        )
        msg["openai_status"] = "ok"
    except Exception as e:
        msg["verified"] = False
        msg["corrected_intent"] = msg["intent"]
        msg["openai_status"] = f"error: {e}"
    return msg

async def verify_batch_openai(batch):
    tasks = [verify_one_openai(msg) for msg in batch]
    return await asyncio.gather(*tasks)

# -------------------------------------------------------
# Main pipeline
# -------------------------------------------------------
async def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} conversation sequences")

    # --- Labeling with Groq ---
    print("🧩 Labeling with Groq (async batches)...")
    labeled_all = []
    start_time = time.perf_counter()
    for i in tqdm(range(0, len(data), BATCH_SIZE)):
        batch = data[i:i + BATCH_SIZE]
        results = await label_batch_groq(batch)
        labeled_all.extend(results)
    elapsed = time.perf_counter() - start_time

    ok_count = sum(1 for x in labeled_all if x["groq_status"] == "ok")
    err_count = len(labeled_all) - ok_count
    print(f"\n⚙️  Groq labeling done:")
    print(f"   ✅ Success: {ok_count} / {len(labeled_all)}")
    print(f"   ❌ Errors: {err_count}")
    print(f"   ⌛ Total time: {elapsed/60:.1f} min\n")

    os.makedirs("data", exist_ok=True)
    with open(GROQ_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(labeled_all, f, indent=2, ensure_ascii=False)
    print(f"✅ Groq-labelled data saved to {GROQ_OUTPUT}")

    # --- Verification with GPT-4o ---
    random.seed(RANDOM_SEED)
    sample_size = max(1, int(len(labeled_all) * VERIFY_FRACTION))
    subset = random.sample(labeled_all, sample_size)
    print(f"🧮 Auditing {sample_size} samples with GPT-4o (async batches)...")
    verified_subset = []
    for i in tqdm(range(0, len(subset), BATCH_SIZE)):
        batch = subset[i:i + BATCH_SIZE]
        verified_results = await verify_batch_openai(batch)
        verified_subset.extend(verified_results)

    disagree = [x for x in verified_subset if not x.get("verified")]
    disagree_rate = len(disagree) / len(verified_subset) if verified_subset else 0
    print(f"\n🧾 Audit complete:")
    print(f"   🔍 Disagreement rate: {disagree_rate*100:.1f}% ({len(disagree)}/{len(verified_subset)})")

    with open(FINAL_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(verified_subset, f, indent=2, ensure_ascii=False)
    print(f"✅ Verified data subset saved to {FINAL_OUTPUT}")


if __name__ == "__main__":
    asyncio.run(main())
