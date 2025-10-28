#!/usr/bin/env python3
"""
Efficient Batched OpenAI Knowledge Distillation Data Preparation Script

Key improvements over concurrent approach:
- True batch processing using OpenAI's batch API
- Significantly reduced API costs (50% discount on batch requests)
- Better rate limit handling
- Simplified processing pipeline
- Built-in retry and error handling
- Optimized async processing with high concurrency (50-100 concurrent requests)
- Fully resumable for both batch and real-time modes
- Incremental saving to prevent data loss
"""

import json
import asyncio
import logging
import os
import math
import re
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import openai
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter

# Load environment variables
load_dotenv()

# Configure logging with debug capability
log_level = logging.DEBUG if os.getenv("DEBUG_LOGPROBS", "false").lower() == "true" else logging.INFO
logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ConversationData:
    """Data structure for conversation with labels and logits"""
    chat_id: str
    sequence: str
    sequence_truncated: str
    gpt4o_label: str
    openai_label: str
    openai_logits: List[float]
    openai_tokens: List[str]
    agreement: bool

class BatchJobStateManager:
    """Manages the state of a batch job by saving the job ID to a file."""
    def __init__(self, state_dir="temp"):
        self.state_dir = state_dir
        self.state_filepath = os.path.join(self.state_dir, "batch_job_state.json")
        os.makedirs(self.state_dir, exist_ok=True)

    def save_job_id(self, job_id: str):
        with open(self.state_filepath, "w") as f:
            json.dump({"batch_id": job_id}, f)
        logger.info(f"Saved batch job state: {job_id}")

    def load_job_id(self) -> Optional[str]:
        if os.path.exists(self.state_filepath):
            with open(self.state_filepath, "r") as f:
                try:
                    data = json.load(f)
                    job_id = data.get("batch_id")
                    if job_id:
                        logger.info(f"Loaded batch job state: {job_id}")
                        return job_id
                except json.JSONDecodeError:
                    return None
        return None

    def clear_state(self):
        if os.path.exists(self.state_filepath):
            os.remove(self.state_filepath)
            logger.info("Cleared batch job state.")

class BatchOpenAILogitsExtractor:
    """Efficient batch processing for OpenAI logits extraction"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.sync_client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.valid_class_tokens = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M'}
        self.state_manager = BatchJobStateManager()
        self.all_valid_tokens_sorted = sorted(list(self.valid_class_tokens))
        
        logger.info(f"✅ Batch OpenAI client initialized")
        logger.info(f"🤖 Model: {model}")
        logger.info(f"📊 Valid class tokens: {self.all_valid_tokens_sorted}")

    def create_classification_prompt(self, conversation_text: str) -> str:
        """Create classification prompt for OpenAI"""
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
        
       Output ONLY the single letter classification (A, B, C, D, E, F, G, H, I, J, K, L, M). No JSON, no explanation, just the letter.

        Classify this message:
        User: {conversation_text}

        Classification:
        """

        return LABEL_PROMPT

    def create_batch_requests(self, conversations: List[Dict]) -> List[Dict]:
        batch_requests = []
        for i, conversation in enumerate(conversations):
            text = conversation.get('sequence_truncated', conversation.get('sequence', ''))
            if not text:
                logger.warning(f"Skipping conversation {conversation.get('chat_id', f'idx_{i}')} - no text")
                continue
            
            prompt = self.create_classification_prompt(text)
            
            request = {
                "custom_id": conversation.get('chat_id', f"conv_{i}"),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 5,
                    "logprobs": True,
                    "top_logprobs": 20
                }
            }
            batch_requests.append(request)
        
        logger.info(f"📦 Created {len(batch_requests)} batch requests")
        return batch_requests

    async def process_conversations(self, conversations: List[Dict], use_batch_api: bool, output_path: str, verification_mode: bool):
        if use_batch_api:
            await self._process_with_batch_api(conversations, output_path, verification_mode)
        else:
            await self._process_with_realtime_api(conversations, output_path, verification_mode, max_concurrent=100)

    async def _process_and_save_record(self, conv_data: ConversationData, output_path: str, verification_mode: bool, file_lock: asyncio.Lock):
        if verification_mode and not conv_data.agreement:
            return
        
        if conv_data.openai_label == "unknown" or not conv_data.openai_logits:
            return

        smoothing = 1e-8
        class_probs = {token: smoothing for token in self.all_valid_tokens_sorted}
        
        for token, log_prob in zip(conv_data.openai_tokens, conv_data.openai_logits):
            if token in class_probs:
                class_probs[token] = math.exp(log_prob)
        
        total_prob = sum(class_probs.values())
        if total_prob > 0:
            class_probs = {k: v/total_prob for k, v in class_probs.items()}
        
        soft_labels = [class_probs[token] for token in self.all_valid_tokens_sorted]
        
        distillation_record = {
            'chat_id': conv_data.chat_id,
            'text': conv_data.sequence_truncated,
            'hard_label': conv_data.gpt4o_label,
            'soft_labels': soft_labels,
            'teacher_prediction': conv_data.openai_label,
            'class_order': self.all_valid_tokens_sorted,
            'teacher_confidence': max(soft_labels),
            'agreement': conv_data.agreement
        }

        async with file_lock:
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(distillation_record, ensure_ascii=False) + '\n')

    async def _process_with_realtime_api(self, conversations: List[Dict], output_path: str, verification_mode: bool, max_concurrent: int = 50):
        logger.info(f"🔄 Using real-time API with {max_concurrent} concurrent requests...")
        semaphore = asyncio.Semaphore(max_concurrent)
        file_lock = asyncio.Lock()

        async def process_and_save_with_limit(conv):
            async with semaphore:
                result = await self._process_single_realtime(conv, conv.get('sequence_truncated', conv.get('sequence', '')))
                if result:
                    await self._process_and_save_record(result, output_path, verification_mode, file_lock)

        tasks = [process_and_save_with_limit(conv) for conv in conversations]
        logger.info(f"📦 Processing {len(tasks)} requests with up to {max_concurrent} concurrent...")
        
        for f in tqdm.asyncio.tqdm.as_completed(tasks, desc="Processing Real-time"):
            await f

        logger.info(f"✅ Successfully processed {len(tasks)} conversations in real-time mode.")

    async def _process_with_batch_api(self, conversations: List[Dict], output_path: str, verification_mode: bool):
        logger.info("📦 Processing with OpenAI Batch API (50% cost savings)...")
        batch_id = self.state_manager.load_job_id()

        if not batch_id:
            batch_requests = self.create_batch_requests(conversations)
            if not batch_requests:
                logger.warning("No requests to send. Aborting batch process.")
                return

            batch_file_path = f"temp/batch_input_{int(time.time())}.jsonl"
            with open(batch_file_path, 'w') as f:
                for request in batch_requests:
                    f.write(json.dumps(request) + '\n')
            
            logger.info("📤 Uploading batch file...")
            with open(batch_file_path, 'rb') as f:
                batch_input_file = self.sync_client.files.create(file=f, purpose="batch")
            
            batch_job = self.sync_client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            batch_id = batch_job.id
            self.state_manager.save_job_id(batch_id)
            os.remove(batch_file_path)

        logger.info(f"⏳ Waiting for batch {batch_id} to complete...")
        while True:
            batch_job = self.sync_client.batches.retrieve(batch_id)
            if batch_job.status == "completed":
                logger.info("✅ Batch completed successfully!")
                break
            elif batch_job.status in ["failed", "cancelled"]:
                logger.error(f"❌ Batch {batch_job.status}: {batch_job.errors}")
                self.state_manager.clear_state()
                raise Exception(f"Batch job {batch_job.status}")
            
            logger.info(f"   Status: {batch_job.status} | Progress: {batch_job.request_counts.completed}/{batch_job.request_counts.total}")
            await asyncio.sleep(60)

        logger.info(f"📥 Downloading results from {batch_job.output_file_id}...")
        result_content = self.sync_client.files.content(batch_job.output_file_id).content.decode('utf-8')
        
        conv_lookup = {conv.get('chat_id', f"conv_{i}"): conv for i, conv in enumerate(conversations)}
        file_lock = asyncio.Lock() # Although not strictly needed here, good practice

        lines = result_content.strip().split('\n')
        for line in tqdm.tqdm(lines, desc="Processing Batch Results"):
            if not line.strip(): continue
            result_item = json.loads(line)
            parsed_data = self.parse_single_batch_result(result_item, conv_lookup)
            if parsed_data:
                await self._process_and_save_record(parsed_data, output_path, verification_mode, file_lock)

        self.state_manager.clear_state()
        logger.info("🧹 Cleaned up batch job state.")

    # Contains the parsing logic from the original script, but for a single item
    def parse_single_batch_result(self, result: Dict, conv_lookup: Dict) -> Optional[ConversationData]:
        """Parses a single result from the OpenAI batch API output."""
        try:
            custom_id = result.get('custom_id')
            conversation = conv_lookup.get(custom_id)
            if not conversation:
                logger.warning(f"No matching conversation for {custom_id}")
                return None
            
            if result.get('error'):
                logger.error(f"API error for {custom_id}: {result['error']}")
                return None
            
            response_data = result.get('response', {})
            choices = response_data.get('body', {}).get('choices', [])
            if not choices:
                logger.warning(f"No choices found for {custom_id}")
                return None
            
            choice = choices[0]
            message = choice.get('message', {})
            content = message.get('content', '').strip()
            
            openai_class = None
            content_clean = content.strip().upper()
            for token in self.valid_class_tokens:
                if token in content_clean:
                    openai_class = token
                    break
            if not openai_class:
                letter_match = re.search(r'\b([A-M])\b', content_clean)
                if letter_match:
                    openai_class = letter_match.group(1)
            
            if not openai_class:
                logger.warning(f"Could not extract classification from: '{content}' for {custom_id}")
                return None
            
            raw_tokens, raw_logits = [], []
            logprobs = choice.get('logprobs')
            if logprobs and logprobs.get('content'):
                for token_data in logprobs['content'][:3]:
                    token_text = token_data.get('token', '').strip().upper()
                    if token_text in self.valid_class_tokens:
                        raw_tokens.append(token_text)
                        raw_logits.append(token_data.get('logprob'))
                    if token_data.get('top_logprobs'):
                        for alt in token_data['top_logprobs']:
                            alt_text = alt.get('token', '').strip().upper()
                            if alt_text in self.valid_class_tokens and alt_text not in raw_tokens:
                                raw_tokens.append(alt_text)
                                raw_logits.append(alt.get('logprob'))
            
            filtered_tokens, filtered_logits = self.filter_and_renormalize_logits(raw_tokens, raw_logits)
            
            if not filtered_tokens:
                filtered_tokens = list(self.all_valid_tokens_sorted)
                uniform_log_prob = math.log(1.0 / len(filtered_tokens))
                filtered_logits = [uniform_log_prob] * len(filtered_tokens)
            
            gpt4o_class = conversation.get('intent', 'unknown')
            agreement = (openai_class == gpt4o_class)
            
            return ConversationData(
                chat_id=custom_id,
                sequence=conversation.get('sequence', ''),
                sequence_truncated=conversation.get('sequence_truncated', ''),
                gpt4o_label=gpt4o_class,
                openai_label=openai_class,
                openai_logits=filtered_logits,
                openai_tokens=filtered_tokens,
                agreement=agreement
            )
        except Exception as e:
            logger.error(f"Error parsing batch result for {result.get('custom_id', 'unknown')}: {e}", exc_info=True)
            return None

    # Contains the logit filtering logic from the original script
    def filter_and_renormalize_logits(self, tokens: List[str], logits: List[float]) -> Tuple[List[str], List[float]]:
        """Filter logits to only include valid class tokens and renormalize"""
        if not tokens or not logits:
            logger.debug(f"Empty tokens or logits: tokens={len(tokens)}, logits={len(logits)}")
            return [], []
        
        # Log raw tokens for debugging
        logger.debug(f"Raw tokens before filtering: {tokens}")
        logger.debug(f"Raw logits before filtering: {logits}")
        
        # Find indices of valid class tokens
        valid_indices = []
        for i, token in enumerate(tokens):
            clean_token = token.strip().upper()  # Normalize to uppercase
            if clean_token in self.valid_class_tokens:
                valid_indices.append(i)
                logger.debug(f"Found valid token '{clean_token}' with logprob {logits[i]}")
        
        if not valid_indices:
            logger.warning(f"No valid class tokens found in: {tokens}")
            return [], []
        
        # Extract valid tokens and logits
        filtered_tokens = [tokens[i].strip().upper() for i in valid_indices]
        filtered_log_probs = [logits[i] for i in valid_indices]
        
        logger.debug(f"Filtered tokens: {filtered_tokens}")
        logger.debug(f"Filtered logprobs: {filtered_log_probs}")
        
        # Convert log probabilities to probabilities and renormalize
        probs = [math.exp(log_prob) for log_prob in filtered_log_probs]
        total_prob = sum(probs)
        
        if total_prob > 0:
            normalized_probs = [prob / total_prob for prob in probs]
            normalized_log_probs = [math.log(prob) for prob in normalized_probs]
            logger.debug(f"Normalized probabilities: {normalized_probs}")
        else:
            logger.warning("Total probability is zero, using uniform distribution")
            uniform_prob = 1.0 / len(filtered_tokens)
            normalized_log_probs = [math.log(uniform_prob)] * len(filtered_tokens)
        
        return filtered_tokens, normalized_log_probs

    # Contains the single real-time processing logic from the original script
    async def _process_single_realtime(self, conversation: Dict, text: str, max_retries: int = 3) -> Optional[ConversationData]:
        """Process single conversation with real-time API with retry logic"""
        
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
                    timeout=30.0  # Add timeout to prevent hanging
                )
                
                # Parse response (similar to batch parsing)
                if not response.choices:
                    logger.warning(f"No choices in response for {conversation.get('chat_id', 'unknown')}")
                    return None
                
                content = response.choices[0].message.content.strip().upper()
                
                # Extract classification - simplified for single letter response
                openai_class = None
                for token in self.valid_class_tokens:
                    if token in content:
                        openai_class = token
                        break
                
                if not openai_class:
                    letter_match = re.search(r'\b([A-F]|[XYZ])\b', content)
                    if letter_match:
                        openai_class = letter_match.group(1)
                
                if not openai_class:
                    logger.warning(f"Could not extract classification from: '{content}'")
                    return None
                
                # Extract logprobs - Fixed to handle OpenAI's actual response format  
                raw_tokens = []
                raw_logits = []
                
                if (response.choices[0].logprobs and 
                    response.choices[0].logprobs.content and
                    len(response.choices[0].logprobs.content) > 0):
                    
                    # OpenAI returns logprobs for each token in the content array
                    first_token_logprobs = response.choices[0].logprobs.content[0]
                    
                    # The top_logprobs contains ALL alternatives including the chosen one
                    if first_token_logprobs.top_logprobs:
                        for alt_token in first_token_logprobs.top_logprobs:
                            token_text = alt_token.token.strip().upper()
                            # Only collect tokens that could be valid class labels
                            if token_text in self.valid_class_tokens:
                                raw_tokens.append(token_text)
                                raw_logits.append(alt_token.logprob)
                    else:
                        # Fallback: use just the chosen token if no top_logprobs
                        token_text = first_token_logprobs.token.strip().upper()
                        if token_text in self.valid_class_tokens:
                            raw_tokens.append(token_text)
                            raw_logits.append(first_token_logprobs.logprob)
                
                filtered_tokens, filtered_logits = self.filter_and_renormalize_logits(raw_tokens, raw_logits)
                
                if not filtered_tokens:
                    filtered_tokens = list(self.valid_class_tokens)
                    uniform_log_prob = math.log(1.0 / len(filtered_tokens))
                    filtered_logits = [uniform_log_prob] * len(filtered_tokens)
                
                gpt4o_class = conversation.get('intent', conversation.get('label', conversation.get('class', 'unknown')))
                agreement = (openai_class == gpt4o_class)
                
                return ConversationData(
                    chat_id=conversation.get('chat_id', f"unknown_{hash(text) % 10000}"),
                    sequence=conversation.get('sequence', text),
                    sequence_truncated=text,
                    gpt4o_label=gpt4o_class,
                    openai_label=openai_class,
                    openai_logits=filtered_logits,
                    openai_tokens=filtered_tokens,
                    agreement=agreement
                )
            
            except openai.RateLimitError as e:
                wait_time = min(2 ** attempt, 8)  # Exponential backoff, max 8 seconds
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

def analyze_results_from_file(filepath: str, confusion_matrix_path: str):
    """Loads results from a .jsonl file and generates summary analysis."""
    if not os.path.exists(filepath):
        logger.warning(f"Result file not found at {filepath}, skipping analysis.")
        return

    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    logger.info(f"\n--- Final Analysis of {len(results)} Records ---")
    agreements = sum(1 for r in results if r['agreement'])
    agreement_rate = agreements / len(results) if results else 0
    logger.info(f"Agreement Rate: {agreement_rate:.2%}")

    # Confusion Matrix
    y_true = [r['hard_label'] for r in results]
    y_pred = [r['teacher_prediction'] for r in results]
    all_labels = sorted(set(y_true + y_pred))
    cm = np.zeros((len(all_labels), len(all_labels)), dtype=int)
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
    for true, pred in zip(y_true, y_pred):
        cm[label_to_idx[true], label_to_idx[pred]] += 1

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=all_labels, yticklabels=all_labels)
    plt.title(f'Confusion Matrix (GPT-4o vs OpenAI)\nAgreement: {agreement_rate:.2%}')
    plt.xlabel('OpenAI Prediction')
    plt.ylabel('GPT-4o Ground Truth')
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    logger.info(f"💾 Confusion matrix saved to {confusion_matrix_path}")
    plt.close()

def get_processed_ids(filepath: str) -> set:
    processed_ids = set()
    if not os.path.exists(filepath):
        return processed_ids
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try: processed_ids.add(json.loads(line)['chat_id'])
                except (json.JSONDecodeError, KeyError): continue
    return processed_ids

async def main():
    """Main execution function"""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY: logger.error("OPENAI_API_KEY not found"); return

    LOGPROBS_INPUT_FILE = os.getenv("LOGPROBS_INPUT_FILE", "data/gpt4o_labelled.json")
    LOGPROBS_OUTPUT_FILE = os.getenv("LOGPROBS_OUTPUT_FILE", "data/distillation_data_batch.jsonl")
    CONFUSION_MATRIX_PATH = os.getenv("CONFUSION_MATRIX_PATH", "confusion_matrix_batch.png")
    MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    VERIFICATION_MODE = os.getenv("VERIFICATION_MODE", "true").lower() == "true"
    USE_BATCH_API = os.getenv("USE_BATCH_API", "true").lower() == "true"

    logger.info("🚀 Optimized OpenAI Knowledge Distillation Processor")
    # ... (logging config)

    try:
        with open(LOGPROBS_INPUT_FILE, 'r') as f: all_conversations = json.load(f)
    except FileNotFoundError: logger.error(f"Input file {LOGPROBS_INPUT_FILE} not found"); return

    processed_ids = get_processed_ids(LOGPROBS_OUTPUT_FILE)
    conversations_to_process = [c for c in all_conversations if c.get('chat_id') not in processed_ids]
    
    logger.info(f"Found {len(all_conversations)} total conversations, {len(processed_ids)} already processed.")
    logger.info(f"{len(conversations_to_process)} conversations remaining to process.")

    if not conversations_to_process:
        logger.info("✅ No new conversations to process.")
    else:
        processor = BatchOpenAILogitsExtractor(OPENAI_API_KEY, model=MODEL)
        await processor.process_conversations(conversations_to_process, USE_BATCH_API, LOGPROBS_OUTPUT_FILE, VERIFICATION_MODE)

    logger.info("\n--- Starting Final Analysis ---")
    analyze_results_from_file(LOGPROBS_OUTPUT_FILE, CONFUSION_MATRIX_PATH)
    logger.info("✅ All tasks complete!")

if __name__ == "__main__":
    try:
        import tqdm
    except ImportError:
        print("Please install tqdm: pip install tqdm")
    asyncio.run(main())
