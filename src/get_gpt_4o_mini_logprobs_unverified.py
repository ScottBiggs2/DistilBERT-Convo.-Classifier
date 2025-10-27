#!/usr/bin/env python3
"""
OpenAI GPT-4o-mini Single-Pass Labeling and Logprob Extraction
Directly adapted from working batch script - no verification logic
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

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ConversationData:
    """Data structure for conversation with labels and logits"""
    chat_id: str
    sequence: str
    sequence_truncated: str
    predicted_label: str
    logits: List[float]
    tokens: List[str]
    confidence: float

class OpenAILabeler:
    """Single-pass labeling and logits extraction with OpenAI"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        self.valid_class_tokens = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N'}
        
        logger.info(f"✅ OpenAI client initialized")
        logger.info(f"🤖 Model: {model}")
    
    def create_classification_prompt(self, conversation_text: str) -> str:
        """Create classification prompt for OpenAI"""
        INTENT_CATEGORIES_LIST = """
        A - academic_help – Students getting help with homework, assignments, tests, or studying. Key indicators: multiple problems/questions in a row, test/quiz format (multiple choice, true/false, select correct answer), textbook-style questions, requests for step-by-step solutions or translations, academic subject matter (math, science, world languages, history, etc.) in a learning context, asking for explanations of academic concepts. Use this even if not explicitly stated as homework
        B - personal_writing_or_communication – Draft, edit, or improve personal/professional emails, messages, social media posts, letters, or workplace communications. The focus is on REAL correspondence to actual people (boss, colleague, client, friend)
        C - writing_and_editing – Create, edit, or improve nonfiction or instructional writing: essays, reports, arguments, articles, blog posts, or educational materials (lesson plans, assignments, summaries). If the focus is logic, structure, or conveying factual information, consider using this category.
        D - creative_writing_and_role_play – Create poems, stories, fictional narratives, scripts, dialogues, or character-based roleplays. Look for tone, emotion, or imaginative context.If the writing involves characters, world-building, roleplay, sci-fi or fantasy, or other storytelling, consider using this category.
        E - general_guidance_and_info – Provide step-by-step guidance, practical advice, or factual information about how or why something works. Combines procedural "how-to" help with general knowledge or curiosity.
        F - programming_and_data_analysis – Write or debug code or work with data/programming tools. Covers technical problem solving in computing, IT, or analytics contexts.
        G - creative_ideation – Generate new ideas, brainstorm concepts, discover new topics or related resources, or create names/slogans. 
        H - purchasable_products – Ask about products, services, or prices. 
        I - greetings_and_chitchat – Small talk or casual chat, asking about the assistant's day, 
        J - relationships_and_personal_reflection – Discuss emotions, relationships, or introspection. Typically but not strictly non-sexual content. 
        K - media_generation_or_analysis – Create, edit, analyze, or retrieve visual/audio/media content (images, photos, videos). 
        L - other – if there is no indication of what the user wants or if there is an intent that is not listed above; should be rare. e.g. suspicious requests, attempts to extract sensitive information.
        M - other_obscene_or_illegal - if the user is making obscene or illegal requests (including violence, drugs, bigotry, hate speech, etc); should be rare.
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

        Output ONLY the single letter classification (A, B, C, D, E, F, G, H, I, J, K, L, M). No JSON, no explanation, just the letter.

        Classify this message:
        User: {conversation_text}

        Classification:
        """

        return LABEL_PROMPT
    
    def filter_and_renormalize_logits(self, tokens: List[str], logits: List[float]) -> Tuple[List[str], List[float]]:
        """Filter logits to only include valid class tokens and renormalize"""
        if not tokens or not logits:
            return [], []
        
        valid_indices = []
        for i, token in enumerate(tokens):
            clean_token = token.strip().upper()
            if clean_token in self.valid_class_tokens:
                valid_indices.append(i)
        
        if not valid_indices:
            return [], []
        
        filtered_tokens = [tokens[i].strip().upper() for i in valid_indices]
        filtered_log_probs = [logits[i] for i in valid_indices]
        
        probs = [math.exp(log_prob) for log_prob in filtered_log_probs]
        total_prob = sum(probs)
        
        if total_prob > 0:
            normalized_probs = [prob / total_prob for prob in probs]
            normalized_log_probs = [math.log(prob) for prob in normalized_probs]
        else:
            uniform_prob = 1.0 / len(filtered_tokens)
            normalized_log_probs = [math.log(uniform_prob)] * len(filtered_tokens)
        
        return filtered_tokens, normalized_log_probs
    
    async def process_with_realtime_api(self, conversations: List[Dict], max_concurrent: int = 50) -> List[ConversationData]:
        """Process with real-time API using concurrent batching with rate limiting"""

        logger.info(f"🔄 Using real-time API with {max_concurrent} concurrent requests...")
        
        # Semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_limit(conv, text):
            async with semaphore:
                return await self._process_single_realtime(conv, text)
        
        # Prepare tasks
        task_inputs = []
        for conv in conversations:
            text = conv.get('formatted_chat', conv.get('sequence_truncated', conv.get('sequence', '')))
            if text:
                task_inputs.append((conv, text))
        
        logger.info(f"📦 Processing {len(task_inputs)} requests...")
        
        # Launch tasks gradually with delay to avoid burst
        results = []
        start_time = time.time()
        active_tasks = set()
        delay_between_starts = 0.2  # INCREASED to 200ms between starting each request
        last_start_time = 0
        
        for idx, (conv, text) in enumerate(task_inputs):
            # Enforce minimum delay between starts
            now = time.time()
            time_since_last_start = now - last_start_time
            if time_since_last_start < delay_between_starts:
                await asyncio.sleep(delay_between_starts - time_since_last_start)
            
            # Create and start task
            task = asyncio.create_task(process_with_limit(conv, text))
            active_tasks.add(task)
            last_start_time = time.time()
            
            # Collect completed tasks
            done_tasks = {t for t in active_tasks if t.done()}
            for done_task in done_tasks:
                try:
                    result = await done_task
                    if result:
                        results.append(result)
                except Exception:
                    pass
            active_tasks -= done_tasks
            
            # Progress update
            if (idx + 1) % 50 == 0 or (idx + 1) == len(task_inputs):
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                logger.info(f"   Progress: {idx + 1}/{len(task_inputs)} ({(idx + 1)/len(task_inputs)*100:.1f}%) | Rate: {rate:.1f} req/s")
        
        # Wait for remaining tasks
        if active_tasks:
            remaining_results = await asyncio.gather(*active_tasks, return_exceptions=True)
            for result in remaining_results:
                if isinstance(result, ConversationData):
                    results.append(result)
        
        logger.info(f"✅ Successfully processed {len(results)}/{len(task_inputs)} conversations")
        return results
    
    async def _process_single_realtime(self, conversation: Dict, text: str, max_retries: int = 3) -> Optional[ConversationData]:
        """Process single conversation - COPIED FROM REFERENCE SCRIPT"""
        
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
                    timeout=30.0
                )
                
                if not response.choices:
                    return None
                
                content = response.choices[0].message.content.strip().upper()
                
                # Extract classification
                predicted_label = None
                for token in self.valid_class_tokens:
                    if token in content:
                        predicted_label = token
                        break
                
                if not predicted_label:
                    letter_match = re.search(r'\b([A-N])\b', content)
                    if letter_match:
                        predicted_label = letter_match.group(1)
                
                if not predicted_label:
                    return None
                
                # Extract logprobs - EXACT COPY FROM REFERENCE
                raw_tokens = []
                raw_logits = []
                
                if (response.choices[0].logprobs and 
                    response.choices[0].logprobs.content and
                    len(response.choices[0].logprobs.content) > 0):
                    
                    first_token_logprobs = response.choices[0].logprobs.content[0]
                    
                    if first_token_logprobs.top_logprobs:
                        for alt_token in first_token_logprobs.top_logprobs:
                            token_text = alt_token.token.strip().upper()
                            if token_text in self.valid_class_tokens:
                                raw_tokens.append(token_text)
                                raw_logits.append(alt_token.logprob)
                    else:
                        token_text = first_token_logprobs.token.strip().upper()
                        if token_text in self.valid_class_tokens:
                            raw_tokens.append(token_text)
                            raw_logits.append(first_token_logprobs.logprob)
                
                filtered_tokens, filtered_logits = self.filter_and_renormalize_logits(raw_tokens, raw_logits)
                
                if not filtered_tokens:
                    filtered_tokens = list(self.valid_class_tokens)
                    uniform_log_prob = math.log(1.0 / len(filtered_tokens))
                    filtered_logits = [uniform_log_prob] * len(filtered_tokens)
                
                confidence = max([math.exp(lp) for lp in filtered_logits]) if filtered_logits else 0.0
                
                return ConversationData(
                    chat_id=conversation.get('chat_id', f"unknown_{hash(text) % 10000}"),
                    sequence=conversation.get('sequence', text),
                    sequence_truncated=text,
                    predicted_label=predicted_label,
                    logits=filtered_logits,
                    tokens=filtered_tokens,
                    confidence=confidence
                )
            
            except openai.RateLimitError as e:
                wait_time = min(2 ** attempt, 8)
                logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt+1}/{max_retries}")
                await asyncio.sleep(wait_time)
                
            except openai.APITimeoutError as e:
                logger.warning(f"Timeout on attempt {attempt+1}/{max_retries}: {e}")
                if attempt == max_retries - 1:
                    return None
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing conversation (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return None
                await asyncio.sleep(0.5)
        
        return None

def create_label_distribution_chart(results: List[ConversationData], save_path: str):
    label_counts = Counter([r.predicted_label for r in results])
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
    valid_results = [r for r in results if r.predicted_label and len(r.logits) > 0]
    
    all_valid_tokens = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
    distillation_data = []
    
    for conv in valid_results:
        if conv.logits and conv.tokens:
            smoothing = 1e-8
            class_probs = {token: smoothing for token in all_valid_tokens}
            
            for token, log_prob in zip(conv.tokens, conv.logits):
                if token in class_probs:
                    class_probs[token] = math.exp(log_prob)
            
            total_prob = sum(class_probs.values())
            if total_prob > 0:
                class_probs = {k: v/total_prob for k, v in class_probs.items()}
            
            soft_labels = [class_probs[token] for token in all_valid_tokens]
            
            distillation_data.append({
                'chat_id': conv.chat_id,
                'text': conv.sequence_truncated,
                'hard_label': conv.predicted_label,
                'soft_labels': soft_labels,
                'class_order': all_valid_tokens,
                'confidence': conv.confidence
            })
    
    output_data = {
        'metadata': {
            'total_processed': len(results),
            'valid_samples': len(valid_results),
            'teacher_model': 'gpt-4o-mini',
            'class_order': all_valid_tokens,
            'distillation_samples': len(distillation_data)
        },
        'distillation_ready': distillation_data
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Saved {len(distillation_data)} training samples")
    
    label_counts = Counter([d['hard_label'] for d in distillation_data])
    logger.info(f"📊 Label distribution:")
    for label in sorted(label_counts.keys()):
        logger.info(f"   {label}: {label_counts[label]} ({label_counts[label]/len(distillation_data)*100:.1f}%)")
    
    return len(distillation_data)

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
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        logger.error("❌ OPENAI_API_KEY not found")
        return
    
    INPUT_FILE = os.getenv("INPUT_FILE", "data/100k_convo_Oct_2025.json")
    OUTPUT_FILE = os.getenv("OUTPUT_FILE", "data/distillation_data_gpt4o_mini.json")
    DISTRIBUTION_CHART = os.getenv("DISTRIBUTION_CHART_PATH", "label_distribution.png")
    MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "5"))
    
    logger.info("🚀 OpenAI GPT-4o-mini Labeler")
    
    try:
        conversations = load_conversations(INPUT_FILE)
    except FileNotFoundError:
        logger.error(f"❌ File not found: {INPUT_FILE}")
        return
    
    if not conversations:
        logger.error("❌ No conversations loaded")
        return
    
    labeler = OpenAILabeler(OPENAI_API_KEY, model=MODEL)
    
    start_time = time.time()
    results = await labeler.process_with_realtime_api(conversations, max_concurrent=MAX_CONCURRENT)
    elapsed_time = time.time() - start_time
    
    if not results:
        logger.error("❌ No results")
        return
    
    logger.info(f"⏱️  Total time: {elapsed_time/60:.1f}m ({len(results)/elapsed_time:.1f} req/s)")
    
    create_label_distribution_chart(results, DISTRIBUTION_CHART)
    save_distillation_data(results, OUTPUT_FILE)
    
    logger.info("✅ Complete!")

if __name__ == "__main__":
    asyncio.run(main())