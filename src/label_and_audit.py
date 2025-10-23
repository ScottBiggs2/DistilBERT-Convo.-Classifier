#!/usr/bin/env python3
"""
Async batched labeling pipeline for pre-truncated conversation sequences.
- Labels each sequence using GPT-4o-mini (OpenAI API)
- Audits a random subset using GPT-5 (OpenAI API)
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
from openai import AsyncOpenAI

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
load_dotenv()

INPUT_FILE = "data/cleaned_sequences.json"
LABELED_OUTPUT = "data/gpt4o_labelled.json"
FINAL_OUTPUT = "data/verified_gpt5.json"

BATCH_SIZE = int(os.getenv("LABEL_BATCH_SIZE", 8))
VERIFY_FRACTION = 0.025
RANDOM_SEED = 42

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------------------------------
# Intent taxonomy
# -------------------------------------------------------
INTENT_CATEGORIES_LIST_SIMPLIFIED = """
→ academic_help – Students getting help with homework, assignments, tests, or studying. Key indicators: multiple problems/questions in a row, test/quiz format (multiple choice, true/false, select correct answer), textbook-style questions, requests for step-by-step solutions, academic subject matter (math, science, history, etc.) in a learning context, asking for explanations of academic concepts. Use this even if not explicitly stated as homework - look for academic question patterns. e.g. "solve this chemistry problem," "select the correct answer for question 3," "explain photosynthesis," "which statement about the Texas Constitution is true?" "what's the pattern in this sequence," "help me with tag questions," "computer networking question"
→ personal_writing_or_communication – Draft, edit, or improve personal/professional emails, messages, letters, or workplace communications. The focus is on REAL correspondence to actual people (boss, colleague, client, friend). e.g. "email my boss about vacation," "improve this message to a client," "write a professional email," "fix grammar in my work message," "polish this letter."
→ writing_and_editing – Write, edit, or improve general content like essays, articles, reports, arguments, summaries, OR create educational materials/content (lesson plans, discussion questions, assignments for others). This includes TEACHERS/EDUCATORS creating content, not students doing homework. NOT for personal messages. e.g. "fix this paragraph," "summarize this article," "write a thesis statement," "edit my blog post," "create discussion questions for my class," "write assignment descriptions," "create a lesson plan."
→ write_fiction – Create poems, stories, or fictional narratives. e.g. "write a poem about space," "create a short story."
→ how_to_advice – Give step-by-step guidance or instructions for accomplishing a task or learning a skill. e.g. "how to bake bread," "career advice for someone with schizophrenia," "create a weekend routine."
→ creative_ideation – Generate new ideas, brainstorm concepts, or create names/slogans. e.g. "podcast ideas," "marketing phrases for my business."
→ translation – Translate text between languages. e.g. "translate to French."
→ computer_programming – Write or debug code. e.g. "SQL query," "fix this Python script."
→ purchasable_products – Ask about products or prices. e.g. "best laptop."
→ cooking_and_recipes – Request recipes or cooking tips. e.g. "lasagna recipe."
→ health_fitness_beauty_or_self_care – Ask about wellness or routines. e.g. "improve sleep," "medical advice about medication."
→ specific_info – Request factual information for general knowledge or curiosity, NOT for homework/studying. If the question looks like it could be from a test, homework, or study session → use academic_help instead. e.g. "who is Marie Curie" (casual interest), "what are fanfiction readers called," "tell me about agriculture" (general interest).
→ greetings_and_chitchat – Small talk or casual chat. e.g. "hi there."
→ relationships_and_personal_reflection – Discuss emotions or relationships. e.g. "I feel anxious."
→ games_and_role_play – Engage in imaginative, entertainment, or character-based roleplay. Look for fictional scenarios, romance themes, unusual character names, or action descriptions in parentheses or asterisks like (*smiles*), (approaches), *bows*. e.g. "pretend to be a wizard," "act as my interviewer," "continue as the queen," "*waves* hello traveler."
→ media_generation_or_analysis – Create, edit, analyze, or retrieve visual/audio/media content (images, photos, videos). e.g. "draw an image," "describe this photo," "combine these two photos," "add a Christmas theme to this image."
→ unclear – if there is no indication of what the user wants; usually this would be a very short prompt or fragmented message.
→ other – if there is an intent that is not listed above; should be pretty rare. e.g. suspicious requests, extracting sensitive information.
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
async def gpt_label_one(item, max_retries=3):
    text = item.get("sequence", "")
    for attempt in range(max_retries):
        try:
            response = await openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": LABEL_PROMPT},
                    {"role": "user", "content": text}
                ],
                temperature=0
            )

            # --- Step 1: log raw response ---
            raw_response = response.choices[0].message.content.strip()
            item["raw_response"] = raw_response[:500]

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
            item["status"] = "ok"
            return item

        except Exception as e:
            print(f"⚠️ GPT-4o-mini attempt {attempt+1}/{max_retries} failed: {e}")
            item["status"] = f"error: {e}"
            await asyncio.sleep(1)

    item["intent"] = "unclear"
    return item


async def label_batch(batch):
    tasks = [gpt_label_one(item) for item in batch]
    return await asyncio.gather(*tasks)


async def verify_one_gpt5(msg):
    text = msg.get("sequence", "")
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": VERIFY_PROMPT},
                {"role": "user", "content": json.dumps({"content": text, "intent": msg["intent"]})}
            ],
        )
        verification = json.loads(response.choices[0].message.content)
        msg["verified"] = verification.get("verified", False)
        msg["corrected_intent"] = (
            msg["intent"]
            if verification.get("corrected_intent") == "same"
            else verification.get("corrected_intent", msg["intent"])
        )
        msg["audit_status"] = "ok"
    except Exception as e:
        msg["verified"] = False
        msg["corrected_intent"] = msg["intent"]
        msg["audit_status"] = f"error: {e}"
    return msg


async def verify_batch(batch):
    tasks = [verify_one_gpt5(msg) for msg in batch]
    return await asyncio.gather(*tasks)


# -------------------------------------------------------
# Main pipeline
# -------------------------------------------------------
async def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} conversation sequences")

    # --- Labeling with GPT-4o-mini ---
    print("🧩 Labeling with GPT-4o-mini (async batches)...")
    labeled_all = []
    start_time = time.perf_counter()
    for i in tqdm(range(0, len(data), BATCH_SIZE)):
        batch = data[i:i + BATCH_SIZE]
        results = await label_batch(batch)
        labeled_all.extend(results)
    elapsed = time.perf_counter() - start_time

    ok_count = sum(1 for x in labeled_all if x["status"] == "ok")
    err_count = len(labeled_all) - ok_count
    print(f"\n⚙️ GPT-4o-mini labeling done:")
    print(f"   ✅ Success: {ok_count} / {len(labeled_all)}")
    print(f"   ❌ Errors: {err_count}")
    print(f"   ⌛ Total time: {elapsed/60:.1f} min\n")

    os.makedirs("data", exist_ok=True)
    with open(LABELED_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(labeled_all, f, indent=2, ensure_ascii=False)
    print(f"✅ GPT-4o-mini labelled data saved to {LABELED_OUTPUT}")

    # --- Verification with GPT-5 ---
    random.seed(RANDOM_SEED)
    sample_size = max(1, int(len(labeled_all) * VERIFY_FRACTION))
    subset = random.sample(labeled_all, sample_size)
    print(f"🧮 Auditing {sample_size} samples with GPT-5 (async batches)...")
    verified_subset = []
    for i in tqdm(range(0, len(subset), BATCH_SIZE)):
        batch = subset[i:i + BATCH_SIZE]
        verified_results = await verify_batch(batch)
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
