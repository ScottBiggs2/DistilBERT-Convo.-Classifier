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

class BatchOpenAILogitsExtractor:
    """Efficient batch processing for OpenAI logits extraction"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", use_batch_api: bool = True):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.sync_client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.use_batch_api = use_batch_api
        self.valid_class_tokens = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N'}
        
        logger.info(f"✅ Batch OpenAI client initialized")
        logger.info(f"🤖 Model: {model}")
        logger.info(f"📦 Batch API: {'ON (50% cost savings)' if use_batch_api else 'OFF'}")
        logger.info(f"📊 Valid class tokens: {sorted(self.valid_class_tokens)}")
    
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
        """Create batch requests for OpenAI batch API"""
        
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
    
    def create_batch_file(self, batch_requests: List[Dict], filename: str) -> str:
        """Create JSONL file for batch processing"""
        
        os.makedirs("temp", exist_ok=True)
        filepath = f"temp/{filename}"
        
        with open(filepath, 'w') as f:
            for request in batch_requests:
                f.write(json.dumps(request) + '\n')
        
        logger.info(f"📄 Batch file created: {filepath}")
        return filepath
    
    def submit_batch_job(self, batch_file_path: str) -> str:
        """Submit batch job to OpenAI"""
        
        logger.info("📤 Uploading batch file...")
        
        # Upload the file
        with open(batch_file_path, 'rb') as f:
            batch_input_file = self.sync_client.files.create(
                file=f,
                purpose="batch"
            )
        
        logger.info(f"📁 File uploaded: {batch_input_file.id}")
        
        # Create batch job
        batch_job = self.sync_client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "Knowledge distillation logits extraction"}
        )
        
        logger.info(f"🚀 Batch job submitted: {batch_job.id}")
        logger.info(f"📊 Status: {batch_job.status}")
        
        return batch_job.id
    
    def wait_for_batch_completion(self, batch_id: str, check_interval: int = 60) -> Dict:
        """Wait for batch job to complete"""
        
        logger.info(f"⏳ Waiting for batch {batch_id} to complete...")
        start_time = time.time()
        
        while True:
            batch_job = self.sync_client.batches.retrieve(batch_id)
            
            elapsed = time.time() - start_time
            logger.info(f"📊 Status: {batch_job.status} | Elapsed: {elapsed/60:.1f}m")
            
            if batch_job.status == "completed":
                logger.info("✅ Batch completed successfully!")
                return batch_job
            elif batch_job.status == "failed":
                logger.error(f"❌ Batch failed: {batch_job.errors}")
                raise Exception(f"Batch job failed: {batch_job.errors}")
            elif batch_job.status == "cancelled":
                logger.error("❌ Batch was cancelled")
                raise Exception("Batch job was cancelled")
            
            # Log progress if available
            if hasattr(batch_job, 'request_counts'):
                counts = batch_job.request_counts
                logger.info(f"   Progress: {counts.completed}/{counts.total} completed")
            
            time.sleep(check_interval)
    
    def download_batch_results(self, batch_job: Dict) -> List[Dict]:
        """Download and parse batch results with raw content inspection"""
        
        if not batch_job.output_file_id:
            raise Exception("No output file ID in completed batch job")
        
        logger.info(f"📥 Downloading results from {batch_job.output_file_id}...")
        
        # Download the results file
        result_file = self.sync_client.files.content(batch_job.output_file_id)
        result_content = result_file.content.decode('utf-8')
        
        # Log raw content for debugging (first few lines)
        lines = result_content.strip().split('\n')
        logger.info(f"📄 Raw JSONL content preview ({len(lines)} lines total):")
        for i, line in enumerate(lines[:3]):  # Show first 3 lines
            if line.strip():
                logger.info(f"   Line {i+1}: {line[:200]}...")
        
        # Parse JSONL results
        results = []
        for line_num, line in enumerate(lines, 1):
            if line.strip():
                try:
                    result = json.loads(line)
                    results.append(result)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse line {line_num}: {e}")
                    logger.error(f"Problematic line: {line[:200]}...")
        
        logger.info(f"📊 Downloaded {len(results)} results")
        return results
    
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
    
    def parse_batch_results(self, batch_results: List[Dict], conversations: List[Dict]) -> List[ConversationData]:
        """Parse batch results into ConversationData objects with flexible response handling"""
        
        logger.info("🔍 Parsing batch results...")
        
        # Create lookup for conversations by chat_id
        conv_lookup = {conv.get('chat_id', f"conv_{i}"): conv for i, conv in enumerate(conversations)}
        
        parsed_results = []
        parsing_errors = []
        
        for result_idx, result in enumerate(batch_results):
            try:
                custom_id = result.get('custom_id', f'unknown_{result_idx}')
                conversation = conv_lookup.get(custom_id)
                
                if not conversation:
                    logger.warning(f"No matching conversation for {custom_id}")
                    continue
                
                # Check for top-level error
                if result.get('error'):
                    error_msg = f"API error for {custom_id}: {result['error']}"
                    logger.error(error_msg)
                    parsing_errors.append(error_msg)
                    continue
                
                # Handle different possible response structures
                response_data = None
                choices = None
                
                # Try multiple possible response structures
                if 'response' in result:
                    response_data = result['response']
                    if isinstance(response_data, dict):
                        # Try response.body.choices first (batch API format)
                        if 'body' in response_data and isinstance(response_data['body'], dict):
                            choices = response_data['body'].get('choices', [])
                        # Fallback to response.choices (direct format)
                        if not choices:
                            choices = response_data.get('choices', [])
                else:
                    # Direct format - choices at top level
                    choices = result.get('choices', [])
                
                if not choices:
                    error_msg = f"No choices found for {custom_id}. Available keys: {list(result.keys())}"
                    if response_data:
                        error_msg += f". Response keys: {list(response_data.keys())}"
                    logger.warning(error_msg)
                    parsing_errors.append(error_msg)
                    continue
                
                choice = choices[0]
                message = choice.get('message', {})
                content = message.get('content', '').strip()
                
                # Extract classification - simplified for single letter response
                openai_class = None
                content_clean = content.strip().upper()
                
                # Look for our valid class tokens in the response
                for token in self.valid_class_tokens:
                    if token in content_clean:
                        openai_class = token
                        break
                
                # Fallback: try regex for single letters
                if not openai_class:
                    letter_match = re.search(r'\b([A-F]|[XYZ])\b', content_clean)
                    if letter_match:
                        openai_class = letter_match.group(1)
                
                if not openai_class:
                    logger.warning(f"Could not extract classification from: '{content}' for {custom_id}")
                    continue
                
                # Extract logprobs - Simplified for single letter responses
                raw_tokens = []
                raw_logits = []
                
                logprobs = choice.get('logprobs')
                if logprobs and logprobs.get('content'):
                    logger.debug(f"🔍 DEBUG: Processing logprobs for {custom_id}")
                    logger.debug(f"   Content tokens count: {len(logprobs['content'])}")
                    
                    # Since we now expect single letter responses, check first few tokens
                    for token_idx, token_data in enumerate(logprobs['content'][:3]):  # Check first 3 tokens
                        token_text = token_data.get('token', '').strip().upper()
                        token_logprob = token_data.get('logprob')
                        
                        logger.debug(f"   Token {token_idx}: '{token_text}' -> {token_logprob}")
                        
                        # Check if this token IS a class label (exact match)
                        if token_text in self.valid_class_tokens:
                            logger.debug(f"   ✅ Found exact class token: {token_text}")
                            raw_tokens.append(token_text)
                            raw_logits.append(token_logprob)
                        
                        # Also get alternative tokens from top_logprobs at this position
                        if token_data.get('top_logprobs'):
                            for alt in token_data['top_logprobs']:
                                alt_text = alt.get('token', '').strip().upper()
                                alt_logprob = alt.get('logprob')
                                logger.debug(f"     Alt: '{alt_text}' -> {alt_logprob}")
                                
                                if alt_text in self.valid_class_tokens and alt_text not in raw_tokens:
                                    logger.debug(f"   ✅ Found alt class token: {alt_text}")
                                    raw_tokens.append(alt_text)
                                    raw_logits.append(alt_logprob)
                    
                    logger.debug(f"   Final collected tokens: {raw_tokens}")
                    logger.debug(f"   Final collected logprobs: {raw_logits}")
                    
                    if not raw_tokens:
                        # Log all tokens for debugging
                        all_tokens = [t.get('token', '') for t in logprobs['content']]
                        logger.warning(f"   No class tokens found! All tokens: {all_tokens}")
                else:
                    logger.warning(f"No logprobs data found for {custom_id}")
                
                # Filter and renormalize logits
                filtered_tokens, filtered_logits = self.filter_and_renormalize_logits(raw_tokens, raw_logits)
                
                # Fallback if no valid logits
                if not filtered_tokens:
                    filtered_tokens = list(self.valid_class_tokens)
                    uniform_log_prob = math.log(1.0 / len(filtered_tokens))
                    filtered_logits = [uniform_log_prob] * len(filtered_tokens)
                
                # Get original label and check agreement
                gpt4o_class = conversation.get('intent', conversation.get('label', conversation.get('class', 'unknown')))
                agreement = (openai_class == gpt4o_class)
                
                parsed_results.append(ConversationData(
                    chat_id=custom_id,
                    sequence=conversation.get('sequence', ''),
                    sequence_truncated=conversation.get('sequence_truncated', conversation.get('sequence', '')),
                    gpt4o_label=gpt4o_class,
                    openai_label=openai_class,
                    openai_logits=filtered_logits,
                    openai_tokens=filtered_tokens,
                    agreement=agreement
                ))
                
            except Exception as e:
                error_msg = f"Error parsing result {result_idx}: {e}"
                logger.error(error_msg)
                parsing_errors.append(error_msg)
                continue
        
        logger.info(f"✅ Parsed {len(parsed_results)} successful results")
        if parsing_errors:
            logger.warning(f"⚠️ Encountered {len(parsing_errors)} parsing errors")
            # Log first few errors for debugging
            for error in parsing_errors[:3]:
                logger.warning(f"   {error}")
        
        return parsed_results
    
    async def process_with_realtime_api(self, conversations: List[Dict], max_concurrent: int = 50) -> List[ConversationData]:
        """Fallback: process with real-time API using aggressive concurrent batching"""
        
        logger.info(f"🔄 Using real-time API with {max_concurrent} concurrent requests...")
        
        # Semaphore to limit concurrent requests (respects rate limits)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_limit(conv, text):
            async with semaphore:
                return await self._process_single_realtime(conv, text)
        
        # Create ALL tasks at once
        tasks = []
        for conv in conversations:
            text = conv.get('sequence_truncated', conv.get('sequence', ''))
            if text:
                task = process_with_limit(conv, text)
                tasks.append(task)
        
        logger.info(f"📦 Processing {len(tasks)} requests with up to {max_concurrent} concurrent...")
        
        # Execute ALL tasks concurrently with progress tracking
        results = []
        completed = 0
        
        # Use tqdm for progress bar if available, otherwise log progress
        try:
            from tqdm.asyncio import tqdm as tqdm_asyncio
            # tqdm_asyncio.gather doesn't support return_exceptions, so we need to use asyncio.gather
            # and wrap with tqdm manually
            async def gather_with_progress():
                async_iter = asyncio.as_completed(tasks)
                results = []
                with tqdm_asyncio(total=len(tasks), desc="Processing") as pbar:
                    for coro in async_iter:
                        try:
                            result = await coro
                            results.append(result)
                        except Exception as e:
                            results.append(e)
                        pbar.update(1)
                return results
            
            results = await gather_with_progress()
            
        except ImportError:
            # Fallback: manual progress tracking without tqdm
            start_time = time.time()
            for coro in asyncio.as_completed(tasks):
                try:
                    result = await coro
                    results.append(result)
                except Exception as e:
                    results.append(e)
                completed += 1
                if completed % 50 == 0 or completed == len(tasks):
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    logger.info(f"   Progress: {completed}/{len(tasks)} ({completed/len(tasks)*100:.1f}%) | Rate: {rate:.1f} req/s")
        
        # Filter successful results
        successful_results = []
        error_count = 0
        for result in results:
            if isinstance(result, ConversationData):
                successful_results.append(result)
            elif isinstance(result, Exception):
                error_count += 1
                if error_count <= 3:  # Only log first few errors
                    logger.error(f"Task failed: {result}")
        
        if error_count > 0:
            logger.warning(f"⚠️  {error_count} tasks failed")
        
        logger.info(f"✅ Successfully processed {len(successful_results)}/{len(tasks)} conversations")
        return successful_results
    
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
    
    async def process_conversations(self, conversations: List[Dict]) -> List[ConversationData]:
        """Main processing method - uses batch API if available, falls back to real-time"""
        
        # Batch API only makes sense for larger jobs (>100 samples)
        # For 500 samples, real-time with high concurrency is actually faster!
        if self.use_batch_api and len(conversations) > 100:
            try:
                logger.info("📦 Attempting batch API (better for >1000 samples)...")
                return await self._process_with_batch_api(conversations)
            except Exception as e:
                logger.warning(f"Batch API failed: {e}")
                logger.info("🔄 Falling back to optimized real-time API...")
                return await self.process_with_realtime_api(conversations, max_concurrent=50)
        else:
            logger.info("🚀 Using optimized real-time API (best for <1000 samples)")
            # Use higher concurrency for better speed
            max_concurrent = min(100, len(conversations))  # Scale with size
            return await self.process_with_realtime_api(conversations, max_concurrent=max_concurrent)
    
    async def _process_with_batch_api(self, conversations: List[Dict]) -> List[ConversationData]:
        """Process using OpenAI batch API"""
        
        logger.info("📦 Processing with OpenAI Batch API (50% cost savings)...")
        
        # Create batch requests
        batch_requests = self.create_batch_requests(conversations)
        
        # Create batch file
        batch_file = self.create_batch_file(batch_requests, f"batch_{int(time.time())}.jsonl")
        
        # Submit batch job
        batch_id = self.submit_batch_job(batch_file)
        
        # Wait for completion
        batch_job = self.wait_for_batch_completion(batch_id)
        
        # Download and parse results
        batch_results = self.download_batch_results(batch_job)
        parsed_results = self.parse_batch_results(batch_results, conversations)
        
        # Cleanup
        os.remove(batch_file)
        logger.info(f"🧹 Cleaned up temporary file: {batch_file}")
        
        return parsed_results

def create_confusion_matrix(results: List[ConversationData], save_path: Optional[str] = None):
    """Create and display confusion matrix for disagreements"""
    
    y_true = []
    y_pred = []
    
    valid_results = [r for r in results if r.openai_label != "unknown"]
    
    for result in valid_results:
        y_true.append(result.gpt4o_label)
        y_pred.append(result.openai_label)
    
    if not y_true:
        logger.warning("No valid results for confusion matrix")
        return
    
    all_labels = sorted(set(y_true + y_pred))
    cm = np.zeros((len(all_labels), len(all_labels)), dtype=int)
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
    
    for true_label, pred_label in zip(y_true, y_pred):
        true_idx = label_to_idx[true_label]
        pred_idx = label_to_idx[pred_label]
        cm[true_idx, pred_idx] += 1
    
    total_samples = len(valid_results)
    agreements = sum(1 for r in valid_results if r.agreement)
    agreement_rate = agreements / total_samples
    
    logger.info(f"📊 Confusion Matrix Analysis:")
    logger.info(f"   Total samples: {total_samples}")
    logger.info(f"   Agreements: {agreements} ({agreement_rate:.2%})")
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=all_labels, yticklabels=all_labels)
    plt.title(f'Confusion Matrix: GPT-4o vs OpenAI\nAgreement Rate: {agreement_rate:.2%}')
    plt.xlabel('OpenAI Prediction')
    plt.ylabel('GPT-4o Ground Truth')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"💾 Confusion matrix saved to {save_path}")
    
    plt.close()  # Close to prevent blocking

def save_distillation_data(results: List[ConversationData], output_path: str, verification_mode: bool = True):
    """Save the processed data for knowledge distillation"""
    
    if verification_mode:
        valid_results = [r for r in results if r.agreement and r.openai_label != "unknown" and len(r.openai_logits) > 0]
        logger.info(f"🔍 Verification mode: Keeping only {len(valid_results)}/{len(results)} agreed samples")
    else:
        valid_results = [r for r in results if r.openai_label != "unknown" and len(r.openai_logits) > 0]
        logger.info(f"📊 All-data mode: Keeping {len(valid_results)}/{len(results)} samples with valid predictions")
    
    all_valid_tokens = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
    distillation_data = []
    
    for conv in valid_results:
        if conv.openai_logits and conv.openai_tokens:
            smoothing = 1e-8
            class_probs = {token: smoothing for token in all_valid_tokens}
            
            for token, log_prob in zip(conv.openai_tokens, conv.openai_logits):
                if token in class_probs:
                    class_probs[token] = math.exp(log_prob)
            
            total_prob = sum(class_probs.values())
            if total_prob > 0:
                class_probs = {k: v/total_prob for k, v in class_probs.items()}
            
            soft_labels = [class_probs[token] for token in all_valid_tokens]
            
            distillation_data.append({
                'chat_id': conv.chat_id,
                'text': conv.sequence_truncated,
                'hard_label': conv.gpt4o_label,
                'soft_labels': soft_labels,
                'teacher_prediction': conv.openai_label,
                'class_order': all_valid_tokens,
                'teacher_confidence': max(soft_labels),
                'agreement': conv.agreement
            })
    
    output_data = {
        'metadata': {
            'total_processed': len(results),
            'valid_responses': len(valid_results),
            'agreements': len([r for r in valid_results if r.agreement]),
            'agreement_rate': len([r for r in valid_results if r.agreement]) / len(results) if results else 0,
            'verification_mode': verification_mode,
            'teacher_model': 'openai_batch',
            'class_order': all_valid_tokens,
            'processing_method': 'batch_api_with_realtime_fallback',
            'distillation_samples': len(distillation_data)
        },
        'distillation_ready': distillation_data
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Saved {len(distillation_data)} training samples to {output_path}")
    return len(distillation_data)

def load_gpt4o_data(filepath: str) -> List[Dict]:
    """Load GPT-4o labeled conversations from JSON"""
    
    logger.info(f"Loading GPT-4o data from {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        conversations = data
    else:
        conversations = [data]
    
    logger.info(f"Loaded {len(conversations)} conversations")
    return conversations

async def main():
    """Main execution function"""
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not found in environment variables")
        return
    
    LOGPROBS_INPUT_FILE = os.getenv("LOGPROBS_INPUT_FILE", "data/gpt5_labelled.json")
    LOGPROBS_OUTPUT_FILE = os.getenv("LOGPROBS_OUTPUT_FILE", "data/distillation_data_batch.json")
    CONFUSION_MATRIX_PATH = os.getenv("CONFUSION_MATRIX_PATH", "confusion_matrix_batch.png")
    MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    VERIFICATION_MODE = os.getenv("VERIFICATION_MODE", "true").lower() == "true"
    USE_BATCH_API = os.getenv("USE_BATCH_API", "false").lower() == "true"  # Default to false for <1000 samples
    
    logger.info("🚀 Optimized OpenAI Knowledge Distillation Processor")
    logger.info(f"🤖 Model: {MODEL}")
    logger.info(f"📂 Input: {LOGPROBS_INPUT_FILE}")
    logger.info(f"📂 Output: {LOGPROBS_OUTPUT_FILE}")
    logger.info(f"📦 Batch API: {'ON (50% cost savings, ~5-30min wait)' if USE_BATCH_API else 'OFF (real-time async, ~2-3min for 500 samples)'}")
    logger.info(f"🔍 Verification mode: {'ON (agreements only)' if VERIFICATION_MODE else 'OFF (all valid samples)'}")
    
    # Load data
    try:
        conversations = load_gpt4o_data(LOGPROBS_INPUT_FILE)
    except FileNotFoundError:
        logger.error(f"Input file {LOGPROBS_INPUT_FILE} not found")
        return
    
    if not conversations:
        logger.error("No conversations loaded from input file")
        return
    
    # Estimate cost and time
    base_cost_per_1k = 0.15  # Input tokens for gpt-4o-mini
    batch_discount = 0.5 if USE_BATCH_API else 1.0
    estimated_cost = (len(conversations) * 200 * base_cost_per_1k * batch_discount) / 1000
    
    if USE_BATCH_API:
        estimated_time = "5-30 minutes (batch processing)"
    else:
        estimated_time = f"~{len(conversations) / 200:.1f} minutes (async real-time, 50-100 concurrent)"
    
    logger.info(f"⏱️  Estimated time: {estimated_time}")
    logger.info(f"💰 Estimated cost: ${estimated_cost:.2f}")
    
    # Process conversations
    processor = BatchOpenAILogitsExtractor(
        OPENAI_API_KEY, 
        model=MODEL,
        use_batch_api=USE_BATCH_API
    )
    
    start_time = time.time()
    results = await processor.process_conversations(conversations)
    elapsed_time = time.time() - start_time
    
    if not results:
        logger.error("No results obtained from processing")
        return
    
    # Analysis and saving
    valid_results = [r for r in results if r.openai_label != "unknown"]
    logit_results = [r for r in results if len(r.openai_logits) > 0]
    
    logger.info(f"📊 Processing Results:")
    logger.info(f"   Total processed: {len(results)}")
    logger.info(f"   Valid predictions: {len(valid_results)}")
    logger.info(f"   With logits: {len(logit_results)}")
    logger.info(f"   Time elapsed: {elapsed_time:.1f}s ({len(results)/elapsed_time:.1f} req/s)")
    
    create_confusion_matrix(results, CONFUSION_MATRIX_PATH)
    saved_count = save_distillation_data(results, LOGPROBS_OUTPUT_FILE, VERIFICATION_MODE)
    
    logger.info("✅ Optimized processing complete!")
    logger.info(f"💾 {saved_count} samples ready for training")
    logger.info(f"⚡ Processing rate: {len(results)/elapsed_time:.1f} conversations/second")

if __name__ == "__main__":
    try:
        import openai
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install openai matplotlib seaborn numpy")
        exit(1)
    
    asyncio.run(main())