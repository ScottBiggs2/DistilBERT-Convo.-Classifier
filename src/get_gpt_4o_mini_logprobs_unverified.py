#!/usr/bin/env python3
"""
OpenAI GPT-4o-mini Single-Pass Labeling and Logprob Extraction
Fixed version with proper timeout and interrupt handling
"""

import json
import asyncio
import logging
import os
import math
import re
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import openai
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

# Get the project root (one level up from src/)
project_root = Path(__file__).parent.parent
dotenv_path = project_root / ".env"
load_dotenv(dotenv_path=dotenv_path)

# DEBUG: Print what we got
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(f"🔍 .env path: {dotenv_path}")
print(f"🔍 .env exists: {dotenv_path.exists()}")
print(f"🔍 API key found: {OPENAI_API_KEY is not None}")
if OPENAI_API_KEY:
    print(f"🔍 API key starts with: {OPENAI_API_KEY[:10]}")
    print(f"🔍 API key length: {len(OPENAI_API_KEY)}")
else:
    print("❌ API key is None!")


from dotenv import load_dotenv
import os
load_dotenv()
print(os.getenv("OPENAI_API_KEY"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ConversationData:
    """Data structure for conversation with labels and logits"""
    chat_id: str
    sequence: str
    sequence_truncated: str
    gpt4o_label: str
    openai_label: str
    openai_logprobs: List[float]
    openai_tokens: List[str]
    agreement: bool

def save_progress(results: List[ConversationData], temp_output_file: str):
    """Saves the current progress to a temporary file."""
    with open(temp_output_file, 'w', encoding='utf-8') as f:
        json.dump([res.__dict__ for res in results], f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Progress saved to {temp_output_file}")

class OpenAILabeler:
    """Single-pass labeling and logits extraction with OpenAI"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = openai.AsyncOpenAI(api_key=api_key, max_retries=0)
        self.model = model
        self.valid_class_tokens = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2'}
        self.all_valid_tokens_sorted = sorted(list(self.valid_class_tokens))
        logger.info(f"✅ OpenAI client initialized (retries disabled)")
        logger.info(f"🤖 Model: {model}")

    def create_classification_prompt(self, conversation_text: str) -> str:
        """Create classification prompt for OpenAI"""
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

        LABEL_PROMPT = f"""
            You are an internal tool that identifies the primary emotion expressed by a user in their message to an AI chatbot, considering the context of previous messages.

            The messages you are labelling are truncated and preprocessed, and may not follow regular grammar rules smoothly.

            Based on the conversation transcript, select the ONE emotion that best represents the user's primary expressed emotion from their messages. Choose from the categories below, or use `1` for neutral (no strong emotion) or `2` for unknown (cannot determine emotion).

            Emotion categories:
            {EMOTION_CATEGORIES}

            Guidelines for classification:
            * Focus on the USER's emotion, not the content they're discussing. Keep this in mind especially if you suspect the user is doing homework or engaging in roleplay.
            * Consider context: a user asking about sad topics may not themselves be sad. Focus on the content of the text and avoid inferences about the users state of mind.
            * Distinguish between primary and secondary emotions - choose the dominant one
            * 'neutral' (1): calm, matter-of-fact exchanges with no emotional coloring
            * 'unknown' (2): Rare, ambiguous cases where emotion cannot be reliably determined
            * When multiple emotions are present, prioritize the most intense or salient one
            * Pay attention to tone indicators like punctuation (!!!, ???, ...), caps, and emoji

            Output ONLY the single character classification (A-Z, 0, 1, or 2). No JSON, no explanation, just the character.

            Classify this message:
            User: {conversation_text}

            Classification:
            """

        return LABEL_PROMPT

    def filter_and_softmax_logprobs(self, tokens: List[str], logprobs: List[float]) -> Dict[str, float]:
        """Filter logprobs to only include valid class tokens and apply softmax."""
        if not tokens or not logprobs:
            return {}
        
        token_logprob_map = {token.strip().upper(): logprob for token, logprob in zip(tokens, logprobs) if token.strip().upper() in self.valid_class_tokens}

        if not token_logprob_map:
            return {}

        # Apply softmax
        exp_log_probs = [math.exp(lp) for lp in token_logprob_map.values()]
        sum_exp_log_probs = sum(exp_log_probs)
        
        if sum_exp_log_probs > 0:
            probs = {token: math.exp(lp) / sum_exp_log_probs for token, lp in token_logprob_map.items()}
        else:
            probs = {token: 1.0 / len(token_logprob_map) for token in token_logprob_map}
        
        return probs

    async def process_with_realtime_api(self, conversations: List[Dict], max_concurrent: int = 5, progress_interval: int = 100, temp_output_file: str = "temp_results.json") -> List[ConversationData]:
        """Process with real-time API using concurrent batching with rate limiting"""

        logger.info(f"🔄 Using real-time API with {max_concurrent} concurrent requests...")
        
        await asyncio.sleep(2)

        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_limit(conv, text):
            async with semaphore:
                return await self._process_single_realtime(conv, text)
        
        task_inputs = []
        for conv in conversations:
            text = conv.get('formatted_chat', conv.get('sequence_truncated', conv.get('sequence', '')))
            if text:
                task_inputs.append((conv, text))
        
        logger.info(f"📦 Processing {len(task_inputs)} requests...")
        
        results = []
        start_time = time.time()
        active_tasks = set()
        delay_between_starts = 1.0
        last_start_time = 0
        
        try:
            for idx, (conv, text) in enumerate(task_inputs):
                now = time.time()
                time_since_last_start = now - last_start_time
                if time_since_last_start < delay_between_starts:
                    await asyncio.sleep(delay_between_starts - time_since_last_start)
                
                task = asyncio.create_task(process_with_limit(conv, text))
                active_tasks.add(task)
                last_start_time = time.time()
                
                done_tasks = {t for t in active_tasks if t.done()}
                for done_task in done_tasks:
                    try:
                        result = await done_task
                        if result:
                            results.append(result)
                    except Exception:
                        pass
                active_tasks -= done_tasks
                
                if (idx + 1) % progress_interval == 0 or (idx + 1) == len(task_inputs):
                    elapsed = time.time() - start_time
                    rate = (idx + 1) / elapsed if elapsed > 0 else 0
                    logger.info(f"   Progress: {idx + 1}/{len(task_inputs)} ({(idx + 1)/len(task_inputs)*100:.1f}%) | "
                               f"Collected: {len(results)} | Active: {len(active_tasks)} | Rate: {rate:.1f} req/s")
                    save_progress(results, temp_output_file)
        
        except KeyboardInterrupt:
            logger.warning("⚠️  KeyboardInterrupt detected during task launch!")
            raise
        
        if active_tasks:
            logger.info(f"⏳ Waiting for {len(active_tasks)} remaining tasks (max 2 min)...")
            try:
                remaining_results = await asyncio.wait_for(
                    asyncio.gather(*active_tasks, return_exceptions=True),
                    timeout=120.0
                )
                for result in remaining_results:
                    if isinstance(result, ConversationData):
                        results.append(result)
            except asyncio.TimeoutError:
                logger.warning(f"⚠️  Timeout! Cancelling {len(active_tasks)} stuck tasks")
                for task in active_tasks:
                    task.cancel()
        
        logger.info(f"✅ Successfully processed {len(results)}/{len(task_inputs)} conversations")
        return results
    
    async def _process_single_realtime(self, conversation: Dict, text: str, max_retries: int = 2) -> Optional[ConversationData]:
        """Process single conversation with minimal retries"""
        
        prompt = self.create_classification_prompt(text)
        
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=5,
                    logprobs=True,
                    top_logprobs=20,
                    timeout=20.0
                )
                
                if not response.choices:
                    return None
                
                content = response.choices[0].message.content.strip().upper()
                
                openai_class = None
                for token in self.valid_class_tokens:
                    if token in content:
                        openai_class = token
                        break
                
                    if not openai_class:
                        # Match A-Z or 0-2
                        letter_match = re.search(r'\b([A-Z0-2])\b', content)
                        if letter_match:
                            openai_class = letter_match.group(1)
                
                if not openai_class:
                    return None
                
                raw_tokens = []
                raw_logprobs = []
                
                if (response.choices[0].logprobs and 
                    response.choices[0].logprobs.content and
                    len(response.choices[0].logprobs.content) > 0):
                    
                    first_token_logprobs = response.choices[0].logprobs.content[0]
                    
                    if first_token_logprobs.top_logprobs:
                        for alt_token in first_token_logprobs.top_logprobs:
                            token_text = alt_token.token.strip().upper()
                            if token_text in self.valid_class_tokens:
                                raw_tokens.append(token_text)
                                raw_logprobs.append(alt_token.logprob)
                    else:
                        token_text = first_token_logprobs.token.strip().upper()
                        if token_text in self.valid_class_tokens:
                            raw_tokens.append(token_text)
                            raw_logprobs.append(first_token_logprobs.logprob)
                
                prob_map = self.filter_and_softmax_logprobs(raw_tokens, raw_logprobs)
                
                ordered_probs = [prob_map.get(token, 0.0) for token in self.all_valid_tokens_sorted]

                gpt4o_class = conversation.get('emotion', 'unknown')
                agreement = (openai_class == gpt4o_class)

                return ConversationData(
                    chat_id=conversation.get('chat_id', f"unknown_{hash(text) % 10000}"),
                    sequence=conversation.get('sequence', text),
                    sequence_truncated=text,
                    gpt4o_label=gpt4o_class,
                    openai_label=openai_class,
                    openai_logprobs=ordered_probs,
                    openai_tokens=self.all_valid_tokens_sorted,
                    agreement=agreement
                )
            
            except openai.RateLimitError:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return None
                
            except (openai.APITimeoutError, asyncio.TimeoutError):
                return None
                
            except Exception:
                if attempt == max_retries - 1:
                    return None
                await asyncio.sleep(0.5)
        
        return None

def create_label_distribution_chart(results: List[ConversationData], save_path: str):
    label_counts = Counter([r.openai_label for r in results])
    labels = sorted(label_counts.keys())
    counts = [label_counts[label] for label in labels]
    
    plt.figure(figsize=(12, 6))
    plt.bar(labels, counts)
    plt.title(f'Label Distribution (n={len(results)})')
    plt.xlabel('Predicted Label')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.3)
    
    for i, (label, count) in enumerate(zip(labels, counts)):
        plt.text(i, count, str(count), ha='center', va='bottom')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_distillation_data(results: List[ConversationData], output_path: str):
    distillation_data = []
    
    for conv in results:
        distillation_data.append({
            'chat_id': conv.chat_id,
            'text': conv.sequence_truncated,
            'hard_label': conv.gpt4o_label,
            'soft_labels': conv.openai_logprobs,
            'class_order': conv.openai_tokens,
            'teacher_prediction': conv.openai_label,
            'agreement': conv.agreement
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({"distillation_ready": distillation_data}, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Saved {len(distillation_data)} training samples")

def load_conversations(filepath: str) -> List[Dict]:
    logger.info(f"📂 Loading: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        conversations = data
    else:
        conversations = [data]
    
    logger.info(f"✅ Loaded {len(conversations)} conversations")
    return conversations

async def main():
    # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # if not OPENAI_API_KEY:
    #     logger.error("❌ OPENAI_API_KEY not found")
    #     return
    
    # Remove the reload, just use the global
    if not OPENAI_API_KEY:
        logger.error("❌ OPENAI_API_KEY not found")
        return

    INPUT_FILE = os.getenv("INPUT_FILE", "data/cleaned_sequences.json")
    OUTPUT_FILE = os.getenv("OUTPUT_FILE", "data/distillation_data_gpt4o_mini.json")
    DISTRIBUTION_CHART = os.getenv("DISTRIBUTION_CHART_PATH", "label_distribution.png")
    MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "1"))
    TEMP_OUTPUT_FILE = "temp_results.json"

    logger.info("🚀 OpenAI GPT-4o-mini Labeler")
    
    try:
        conversations = load_conversations(INPUT_FILE)
    except FileNotFoundError:
        logger.error(f"❌ File not found: {INPUT_FILE}")
        return
    
    if not conversations:
        logger.error("❌ No conversations loaded")
        return

    results = []
    if os.path.exists(TEMP_OUTPUT_FILE):
        logger.info(f"📂 Loading partial results from {TEMP_OUTPUT_FILE}")
        with open(TEMP_OUTPUT_FILE, 'r', encoding='utf-8') as f:
            try:
                partial_results_data = json.load(f)
                results = [ConversationData(**data) for data in partial_results_data]
                logger.info(f"✅ Loaded {len(results)} partial results")
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"⚠️  Could not decode partial results from {TEMP_OUTPUT_FILE}. Starting from scratch.")
                results = []

    processed_chat_ids = {res.chat_id for res in results}
    conversations_to_process = [conv for conv in conversations if conv.get('chat_id') not in processed_chat_ids]

    if not conversations_to_process:
        logger.info("✅ All conversations have already been processed.")
    else:
        labeler = OpenAILabeler(OPENAI_API_KEY, model=MODEL)
        
        start_time = time.time()
        
        try:
            new_results = await labeler.process_with_realtime_api(
                conversations_to_process, 
                max_concurrent=MAX_CONCURRENT,
                temp_output_file=TEMP_OUTPUT_FILE
            )
            results.extend(new_results)
        except KeyboardInterrupt:
            logger.warning("\n⚠️  KeyboardInterrupt! Saving partial results...")
        finally:
            elapsed_time = time.time() - start_time
            
            if results:
                logger.info(f"⏱️  Total time: {elapsed_time/60:.1f}m ({len(results)/elapsed_time:.1f} req/s)")
                create_label_distribution_chart(results, DISTRIBUTION_CHART)
                save_distillation_data(results, OUTPUT_FILE)
                logger.info("✅ Complete!")
            else:
                logger.error("❌ No results to save")

    if os.path.exists(TEMP_OUTPUT_FILE):
        os.remove(TEMP_OUTPUT_FILE)

if __name__ == "__main__":
    asyncio.run(main())
