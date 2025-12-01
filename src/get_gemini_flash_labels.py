#!/usr/bin/env python3
"""
Async batched labeling pipeline for pre-truncated conversation sequences.
- Labels each sequence using Gemini 2.5 Flash as the primary teacher model
- Logs performance, error rates, raw API responses
- Collects all results in memory and saves as a single JSON file.
- Output will be used with GPT-4o logprobs extractor for knowledge distillation
"""

import os
import json
import asyncio
import time
import re
import logging
from tqdm import tqdm
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
load_dotenv()

INPUT_FILE = "data/cleaned_sequences.json"
# Updated to .json as requested by user
LABELED_OUTPUT = "data/gemini_2.5_flash_labelled.json" 

BATCH_SIZE = int(os.getenv("LABEL_BATCH_SIZE", 32)) # Gemini can handle high concurrency
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 5

# --- Configure Gemini Client ---
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
except TypeError:
    print("❌ GEMINI_API_KEY not found in environment")
    print("Please set your Gemini API key in .env file")
    exit(1)

# Configure safety settings to allow classification of all content
SAFETY_SETTINGS = {
    'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
    'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
    'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
    'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
}

# Configure model for JSON output and deterministic classification
GENERATION_CONFIG = {
    "response_mime_type": "application/json",
    "temperature": 0.0,
}

gemini_model = genai.GenerativeModel(
    'gemini-2.5-flash-lite',
    generation_config=GENERATION_CONFIG,
    safety_settings=SAFETY_SETTINGS
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------------------------------------------
# Intent taxonomy (same as knowledge distillation pipeline)
# -------------------------------------------------------
EMOTION_CATEGORIES = """
    A - admiration
    B - amusement
    C - anger
    D - annoyance
    E - approval
    F - caring
    G - confusion
    H - curiosity
    I - desire
    J - disappointment
    K - disapproval
    L - disgust
    M - embarrassment
    N - excitement
    O - fear
    P - gratitude
    Q - grief
    R - joy
    S - love
    T - nervousness
    U - optimism
    V - pride
    W - realization
    X - relief
    Y - remorse
    Z - sadness
    0 - surprise
    1 - neutral
    2 - unknown
"""

JSON_EXAMPLE = '{"emotion": "<single_character>"}'

LABEL_PROMPT = """
    You are an internal tool that identifies the primary emotion expressed by a user in their message to an AI chatbot, considering the context of previous messages.

    The messages you are labelling are truncated and preprocessed, and may not follow regular grammar rules smoothly.

    Based on the conversation transcript, select the ONE emotion that best represents the user's primary expressed emotion from their messages. Choose from the categories below, or use `1` for neutral (no strong emotion) or `2` for unknown (cannot determine emotion).

    Emotion categories:
    {emotion_categories}

    Guidelines for classification:
    * Focus on the USER's emotion, not the content they're discussing. Keep this in mind especially if you suspect the user is doing homework or engaging in roleplay.
    * Consider context: a user asking about sad topics may not themselves be sad. Focus on the content of the text and avoid inferences about the users state of mind.
    * Distinguish between primary and secondary emotions - choose the dominant one
    * 'neutral' (1): calm, matter-of-fact exchanges with no emotional coloring
    * 'unknown' (2): Rare, ambiguous cases where emotion cannot be reliably determined
    * When multiple emotions are present, prioritize the most intense or salient one
    * Pay attention to tone indicators like punctuation (!!!, ???, ...), caps, and emoji

    Output ONLY the single character classification (A-Z, 0, 1, or 2). No JSON, no explanation, just the character.

    Output ONLY in this JSON format with a SINGLE CHARACTER from the listed emotion categories:
    {json_example}

    Classify this message:
    User: {conversation_text}

    Classification:
    """

# -------------------------------------------------------
# Async helpers
# -------------------------------------------------------
async def gemini_label_one(item, max_retries=MAX_RETRIES):
    """Label a single conversation using Gemini 2.5 Flash"""
    item_copy = item.copy()
    text = item_copy.get("sequence", "")
    prompt_content = LABEL_PROMPT.format(
        emotion_categories=EMOTION_CATEGORIES,  # ✅ Fixed
        json_example=JSON_EXAMPLE,
        conversation_text=text
    )
    
    for attempt in range(max_retries):
        try:
            response = await asyncio.wait_for(
                gemini_model.generate_content_async(prompt_content),
                timeout=REQUEST_TIMEOUT
            )

            raw_response = response.text.strip()
            item_copy["raw_response"] = raw_response[:500]

            # Robust JSON parsing
            try:
                label_json = json.loads(raw_response)
            except json.JSONDecodeError:
                logger.warning(f"JSONDecodeError. Falling back to regex for: {raw_response}")
                match = re.search(r"\{.*\}", raw_response, re.DOTALL)
                if match:
                    try:
                        label_json = json.loads(match.group())
                    except json.JSONDecodeError:
                        # Updated regex for emotion categories: A-Z, 0, 1, 2
                        letter_match = re.search(r'\b([A-Z0-2])\b', raw_response)  # ✅ Fixed
                        label_json = {"emotion": letter_match.group(1) if letter_match else "unclear_parse"}
                else:
                    letter_match = re.search(r'\b([A-Z0-2])\b', raw_response)  # ✅ Fixed
                    label_json = {"emotion": letter_match.group(1) if letter_match else "unclear_parse"}

            item_copy["emotion"] = label_json.get("emotion", "unclear_json")  # ✅ Changed from "intent"
            item_copy["status"] = "ok"
            return item_copy

        except (google_exceptions.ResourceExhausted, google_exceptions.DeadlineExceeded, google_exceptions.InternalServerError) as e:
            logger.warning(f"API Error (Attempt {attempt+1}/{max_retries}): {e}. Retrying after backoff...")
            item_copy["status"] = f"api_error: {e}"
            await asyncio.sleep(2 ** attempt + 0.5)
        except asyncio.TimeoutError:
            logger.warning(f"Request Timeout (Attempt {attempt+1}/{max_retries}). Retrying...")
            item_copy["status"] = "error: timeout"
            await asyncio.sleep(2 ** attempt + 0.5)
        except Exception as e:
            logger.error(f"⚠️ Unhandled error (Attempt {attempt+1}/{max_retries}): {e}")
            item_copy["status"] = f"error: {e}"
            break

    item_copy["emotion"] = "unclear_error"  # ✅ Changed from "intent"
    return item_copy

async def label_batch(batch):
    """Process a batch of conversations concurrently"""
    tasks = [gemini_label_one(item) for item in batch]
    return await asyncio.gather(*tasks)

# -------------------------------------------------------
# Main pipeline
# -------------------------------------------------------
async def main():
    print("🏷️  Gemini 2.5 Flash Conversation Labeling Pipeline")
    print("=" * 50)
    
    # Load data
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"❌ Input file not found: {INPUT_FILE}")
        return
    except json.JSONDecodeError:
        logger.error(f"❌ Could not decode JSON from: {INPUT_FILE}")
        return
        
    total_items = len(data)
    print(f"📂 Loaded {total_items:,} conversation sequences from {INPUT_FILE}")
    
    # Confirm for large datasets
    # if total_items > 1000:
    #     print(f"⚠️  Large dataset detected ({total_items:,} conversations)")
    #     print(f"⚠️  All results will be stored in memory before saving.")
    #     print(f"⚠️  If the script fails, all progress will be lost.")
    #     estimated_time = (total_items / BATCH_SIZE) * 0.5 
    #     print(f"⏱️  Estimated processing time: {estimated_time / 60:.1f} minutes (at {BATCH_SIZE} per batch)")
        
    #     response = input("Continue with Gemini labeling? (y/N): ")
    #     if response.lower() != 'y':
    #         print("❌ Processing cancelled by user")
    #         return
    
    # --- Labeling with Gemini ---
    print(f"\n🧩 Starting Gemini 2.5 Flash labeling (batch size: {BATCH_SIZE})...")
    start_time = time.perf_counter()
    
    # --- This list will hold ALL results in memory ---
    labeled_all = []
    
    # Stats counters
    ok_count = 0
    err_count = 0
    intent_counts = {}

    for i in tqdm.tqdm(range(0, total_items, BATCH_SIZE), desc="Gemini Labeling"):
        batch = data[i:i + BATCH_SIZE]
        results = await label_batch(batch) # results is a list of dicts
        
        # --- Update stats from this batch ---
        for item in results:
            if item["status"] == "ok":
                ok_count += 1
                emotion = item.get("emotion", "unclear")  # ✅ Changed
                intent_counts[emotion] = intent_counts.get(emotion, 0) + 1  # (or rename intent_counts to emotion_counts)
            else:
                err_count += 1
        
        # --- Add batch results to the main list ---
        labeled_all.extend(results)
        
        await asyncio.sleep(0.1) 
    
    elapsed = time.perf_counter() - start_time

    # --- Results summary ---
    print(f"\n📊 Gemini Labeling Results:")
    print(f"   ✅ Successful labels: {ok_count:,} / {total_items:,}")
    print(f"   ❌ Errors: {err_count:,}")
    if total_items > 0:
        print(f"   📈 Success rate: {ok_count/total_items*100:.1f}%")
    print(f"   ⌛ Total time: {elapsed/60:.1f} minutes")
    if elapsed > 0:
        print(f"   🚀 Average rate: {total_items/elapsed:.1f} labels/second")
    
    # Label distribution
    if ok_count > 0:
        print(f"\n🏷️  Label Distribution:")
        for intent, count in sorted(intent_counts.items()):
            percentage = count / ok_count * 100
            print(f"   {intent}: {count:,} ({percentage:.1f}%)")
    
    # --- Save all results to a single JSON file at the end ---
    print(f"\n💾 Saving {len(labeled_all):,} results to {LABELED_OUTPUT}...")
    os.makedirs("data", exist_ok=True)
    with open(LABELED_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(labeled_all, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Gemini labeled data saved to {LABELED_OUTPUT}")
    print(f"🎯 Ready for knowledge distillation pipeline!")

if __name__ == "__main__":
    # Check for required dependencies
    try:
        import google.generativeai
        import tqdm
        import dotenv
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install: pip install google-generativeai tqdm python-dotenv")
        exit(1)
    
    # API key check is handled by genai.configure()
    
    # Run the pipeline
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n❌ Process interrupted by user. No data was saved.")
        exit(1)