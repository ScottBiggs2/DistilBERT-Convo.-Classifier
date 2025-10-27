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
A - academic_help – Students getting help with homework, assignments, tests, or studying. Key indicators: multiple problems/questions in a row, test/quiz format (multiple choice, true/false, select correct answer), textbook-style questions, requests for step-by-step solutions or translations, academic subject matter (math, science, world languages, history, etc.) in a learning context, asking for explanations of academic concepts. Use this even if not explicitly stated as homework
B - personal_writing_or_communication – Draft, edit, or improve personal/professional emails, messages, social media posts, letters, or workplace communications. The focus is on REAL correspondence to actual people (boss, colleague, client, friend)
C - writing_and_editing – Create, edit, or improve nonfiction or instructional writing: essays, reports, arguments, articles, blog posts, or educational materials (lesson plans, assignments, summaries). If the focus is logic, structure, or conveying factual information, consider using this category.
D - creative_writing_and_role_play – Create poems, stories, fictional narratives, scripts, dialogues, or character-based roleplays. Look for tone, emotion, or imaginative context.If the writing involves characters, world-building, roleplay, sci-fi or fantasy, or other storytelling, consider using this category.
E - general_guidance_and_info – Provide step-by-step guidance, practical advice, or factual information about how or why something works. Combines procedural “how-to” help with general knowledge or curiosity.
F - programming_and_data_analysis – Write or debug code or work with data/programming tools. Covers technical problem solving in computing, IT, or analytics contexts.
G - creative_ideation – Generate new ideas, brainstorm concepts, discover new topics or related resources, or create names/slogans. 
H - purchasable_products – Ask about products, services, or prices. 
I - greetings_and_chitchat – Small talk or casual chat, asking about the assistant's day, 
J - relationships_and_personal_reflection – Discuss emotions, relationships, or introspection. Typically but not strictly non-sexual content. 
K - media_generation_or_analysis – Create, edit, analyze, or retrieve visual/audio/media content (images, photos, videos). 
L - other – if there is no indication of what the user wants or if there is an intent that is not listed above; should be rare. e.g. suspicious requests, attempts to extract sensitive information.
M - other_obscene_or_illegal - if the user is making obscene or illegal requests (including violence, drugs, bigotry, hate speech, etc); should be rare.
"""

EXAMPLES_LIST = f"""

A - academic_help:
- "Solve for x: 2x + 3 = 7"
- "How do you calculate the area of a circle?"
- "Explain photosynthesis in simple terms."
- "What is the boiling point of water at sea level?"
- "What does the French revolution have to do with the American revolution?"

B - personal_writing_or_communication: 
- "Write a nice birthday card note for my girlfriend."
- "What should my speech say to Karl at his retirement party?"
- "Help me write a cover letter for a job application."
- "Compose an apology email to my boss."
- "Aide moi `a ´ecrire une lettre `a mon p`ere."

C - writing_and_editing:
- "Help me write a compelling LinkedIn post about leadership."
- "Edit this essay for clarity and grammar."
- "Is my tone in this email too formal?"
- "Summarize the main points of this article."
- "Create an outline for a report on climate change."

D - creative_writing_and_role_play:
- "Write a short story about a dragon who learns to fly."
- "Create a dialogue between a detective and a suspect."
- "Pretend to be a medieval knight on a quest to rescue a princess."
- "Act like Pricess Leia from Star Wars."

E - general_guidance_and_info:
- "How do I turn off my screensaver?"
- "My car won’t start; what should I try?"
- "Comment faire pour me connecter `a mon wifi?"
- "What’s the best way to clean hardwood floors?"
- "How can I replace a flat tire?"

F - programming_and_data_analysis:
- "Write a Python function to sort a list."
- "Debug this JavaScript code for a web form."
- "How do I connect to a database using SQL?"
- "Analyze this dataset to find trends."

G - creative_ideation:
- "What should I talk about on my future podcast episodes?"
- "Give me some themes for a photography project."
- "Necesito ideas para un regalo de aniversario."
- "Brainstorm names for a new coffee shop."
- "What are some unique app ideas for startups?"

H - purchasable_products:
- "What’s the price of the latest iPhone?"
- "Find me a good deal on noise-cancelling headphones."
- "Quels sont les meilleurs ordinateurs portables pour les ´etudiants?"
- "How do I know if a laptop is worth buying?"
- "Show me flights to Tokyo next month."

I - greetings_and_chitchat:
- "How are you doing today?"
- "What’s your favorite movie?"
- "Tell me a joke."
- "Do you like music?"
- "Ciao, parli italiano?"

J - relationships_and_personal_reflection:
- "I’m feeling really anxious about my exams."
- "My partner and I are arguing a lot lately."
- "How can I improve my motivation to work out?"
- "I’m struggling with balancing work and personal life."
- "What should I do if I’m feeling lonely?"

K - media_generation_or_analysis:
- "Write a poem about the ocean."
- "Draw a picture of a flying car being chased by a train."
- "Remove the tourists from this photo of the Eiffel Tower."
- "Describe the scene in this image."

L - other:
- "Tell me a secret about the universe."
- "Can you give me tips on how to manipulate people?"
- "What is the meaning of life?"

M - other_obscene_or_illegal:
- "How can I make a bomb?"
- "How can I hack into someone’s account?"
- "Write a detailed story about a violent crime."
"""

LABEL_PROMPT = f"""
You are an internal tool that classifies a message from a user to an AI chatbot,
based on the context of the previous messages before it.

Based on the contents of this conversation transcript and taking into
account the examples further below as guidance, please select the capability
the user is clearly interested in, or `L` for `other` if it is clear but not in the
list below or if it is hard to tell what the user even wants. 

List of categories:
{INTENT_CATEGORIES_LIST}

Examples of each category, for reference: 
{EXAMPLES_LIST}

Tips and tricks: 
1️⃣ Be careful to distinguish users writing about work for emails, presentations, etc. Words like 'boss', 'meeting', and 'email' will help. 
2️⃣ Be discerning about the flow of the conversation to detect role-play or fictional scenarios, especially when sexual content is involved.

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
            # response = await openai_client.chat.completions.create(
            #     model="gpt-5",  # Using GPT-5 as the primary teacher model
            #     messages=[
            #         {"role": "system", "content": LABEL_PROMPT},
            #         {"role": "user", "content": text}
            #     ]
            # )

            response = await openai_client.chat.completions.create(
                model="gpt-4o",  # Using GPT-5 as the primary teacher model
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