#!/usr/bin/env python3
"""
Async batched labeling pipeline for pre-truncated conversation sequences.
- Labels each sequence using GPT-5 (OpenAI API) as the primary teacher model
- Logs performance, error rates, raw API responses
- Output will be used with GPT-4o logprobs extractor for knowledge distillation
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
LABELED_OUTPUT = "data/gpt5_labelled.json"  # Updated filename for GPT-5 labels

BATCH_SIZE = int(os.getenv("LABEL_BATCH_SIZE", 8))
RANDOM_SEED = 42

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------------------------------
# Intent taxonomy (same as knowledge distillation pipeline)
# -------------------------------------------------------
INTENT_CATEGORIES_LIST = """
A → academic_help – Homework, structured studying, academic problem solving, exam-style questions, translation, academic tutoring, and suspected academic writing work.
B → writing_and_editing – Editing or generating non-fiction text such as essays, summaries, workplace communications (email, slack, etc), translations.
C → creative_content – Fiction, poetry, story creation, creative brainstorming WITHOUT sexual, violent, or other NSFW content.
D → practical_advice – Everyday how-to guidance, general curiosity questions, product recommendations, cooking/baking/gardening, exercise, and other lifestyle info (non-medical).
E → technical_coding – Software development, debugging, coding guidance, ML/AI engineering help, coding completions, and other IT services/questions.
F → personal_support – Emotional expression, relationship concerns, and mental health support. This also includes motivational and self-improvement support. 
"""

NSFW_CATEGORIES = """
X → nsfw_personal – Real-person sexual requests, sexting, explicit intimacy involving real individuals (INCLUDING the assistant).
Y → nsfw_fantasy – Fictional erotic scenarios, roleplay (INCLUDING with the assistant), sexual imagination, fetish elements.
Z → nsfw_other – Pornography media requests, sexual product questions, unclear sexual intent, and other unclear but distinct NSFW content (eg, violence, crime, drug use, etc).
"""

LABEL_PROMPT = f"""
You are a classification model for user chats with LLMs. Assign exactly ONE letter/label to each chat.

Normal categories:
{INTENT_CATEGORIES_LIST}

NSFW override categories:
{NSFW_CATEGORIES}

Tips and tricks: 
1️⃣ Be careful to distinguish users writing about work for emails, presentations, etc. Words like 'boss', 'meeting', and 'email' will help. 
2️⃣ Be discerning about the flow of the conversation to detect role-play or fictional scenarios, especially when sexual content is involved.

Override rules:
1️⃣ If ANY sexual content appears → must use X, Y, or Z.
2️⃣ Use F (personal_support) ONLY when emotional or relational help is primary AND there is no explicit sexual content.
3️⃣ If the message appears to be homework or a test question → use A (academic_help).
4️⃣ If unclear, choose the closest reasonable category — do NOT use multiple labels.

Output ONLY in this JSON format with a SINGLE LETTER from the listed intent categories:
{{"intent": "<single_letter>"}}

Classify this message:
User: {{conversation_text}}
"""

# -------------------------------------------------------
# Async helpers
# -------------------------------------------------------
async def gpt5_label_one(item, max_retries=3):
    """Label a single conversation using GPT-5"""
    text = item.get("sequence", "")
    for attempt in range(max_retries):
        try:
            response = await openai_client.chat.completions.create(
                model="gpt-5",  # Using GPT-5 as the primary teacher model
                messages=[
                    {"role": "system", "content": LABEL_PROMPT},
                    {"role": "user", "content": text}
                ]
            )

            # --- Step 1: log raw response ---
            raw_response = response.choices[0].message.content.strip()
            item["raw_response"] = raw_response[:500]

            # --- Step 2: robust JSON parsing ---
            try:
                label_json = json.loads(raw_response)
            except json.JSONDecodeError:
                # Fallback: extract JSON-like pattern
                match = re.search(r"\{.*\}", raw_response, re.DOTALL)
                if match:
                    try:
                        label_json = json.loads(match.group())
                    except json.JSONDecodeError:
                        # Final fallback: look for single letter
                        letter_match = re.search(r'\b([A-F]|[XYZ])\b', raw_response)
                        if letter_match:
                            label_json = {"intent": letter_match.group(1)}
                        else:
                            label_json = {"intent": "unclear"}
                else:
                    # No JSON found, try to extract single letter
                    letter_match = re.search(r'\b([A-F]|[XYZ])\b', raw_response)
                    if letter_match:
                        label_json = {"intent": letter_match.group(1)}
                    else:
                        label_json = {"intent": "unclear"}

            item["intent"] = label_json.get("intent", "unclear")
            item["status"] = "ok"
            return item

        except Exception as e:
            print(f"⚠️ GPT-5 attempt {attempt+1}/{max_retries} failed: {e}")
            item["status"] = f"error: {e}"
            await asyncio.sleep(2)  # Slightly longer backoff for GPT-5

    item["intent"] = "unclear"
    return item

async def label_batch(batch):
    """Process a batch of conversations concurrently"""
    tasks = [gpt5_label_one(item) for item in batch]
    return await asyncio.gather(*tasks)

# -------------------------------------------------------
# Main pipeline
# -------------------------------------------------------
async def main():
    print("🏷️  GPT-5 Conversation Labeling Pipeline")
    print("=" * 50)
    
    # Load data
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"📂 Loaded {len(data)} conversation sequences from {INPUT_FILE}")
    
    # Estimate costs for GPT-5
    # Note: GPT-5 pricing not yet announced, using placeholder estimates
    estimated_input_tokens = len(data) * 250  # Rough estimate per conversation
    estimated_output_tokens = len(data) * 10  # JSON response
    print(f"💰 Estimated tokens: {estimated_input_tokens:,} input + {estimated_output_tokens:,} output")
    print(f"💡 Note: GPT-5 pricing not yet announced - monitor costs carefully")
    
    # Confirm for large datasets
    if len(data) > 1000:
        print(f"⚠️  Large dataset detected ({len(data):,} conversations)")
        estimated_time = (len(data) / BATCH_SIZE) * 1.5  # Estimate 1.5 sec per batch
        print(f"⏱️  Estimated processing time: {estimated_time / 60:.1f} minutes")
        
        response = input("Continue with GPT-5 labeling? (y/N): ")
        if response.lower() != 'y':
            print("❌ Processing cancelled by user")
            return
    
    # --- Labeling with GPT-5 ---
    print(f"\n🧩 Starting GPT-5 labeling (batch size: {BATCH_SIZE})...")
    labeled_all = []
    start_time = time.perf_counter()
    
    for i in tqdm.tqdm(range(0, len(data), BATCH_SIZE), desc="GPT-5 Labeling"):
        batch = data[i:i + BATCH_SIZE]
        results = await label_batch(batch)
        labeled_all.extend(results)
        
        # Brief pause between batches to be respectful to API
        await asyncio.sleep(0.5)
    
    elapsed = time.perf_counter() - start_time

    # --- Results summary ---
    ok_count = sum(1 for x in labeled_all if x["status"] == "ok")
    err_count = len(labeled_all) - ok_count
    
    print(f"\n📊 GPT-5 Labeling Results:")
    print(f"   ✅ Successful labels: {ok_count:,} / {len(labeled_all):,}")
    print(f"   ❌ Errors: {err_count:,}")
    print(f"   📈 Success rate: {ok_count/len(labeled_all)*100:.1f}%")
    print(f"   ⌛ Total time: {elapsed/60:.1f} minutes")
    print(f"   🚀 Average rate: {len(labeled_all)/elapsed:.1f} labels/second")
    
    # Label distribution
    if ok_count > 0:
        intent_counts = {}
        for item in labeled_all:
            if item["status"] == "ok":
                intent = item.get("intent", "unclear")
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        print(f"\n🏷️  Label Distribution:")
        for intent, count in sorted(intent_counts.items()):
            percentage = count / ok_count * 100
            print(f"   {intent}: {count:,} ({percentage:.1f}%)")
    
    # Save results
    os.makedirs("data", exist_ok=True)
    with open(LABELED_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(labeled_all, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ GPT-5 labeled data saved to {LABELED_OUTPUT}")
    print(f"🎯 Ready for knowledge distillation pipeline!")
    print(f"💡 Next step: Use OpenAI logprobs extractor to find GPT-4o/GPT-5 agreements")

if __name__ == "__main__":
    # Check for required dependencies
    try:
        import openai
        import tqdm
        import dotenv
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install: pip install openai tqdm python-dotenv")
        exit(1)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment")
        print("Please set your OpenAI API key in .env file")
        exit(1)
    
    # Run the pipeline
    asyncio.run(main())