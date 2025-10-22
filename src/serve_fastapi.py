#!/usr/bin/env python3
"""
FastAPI server for serving ONNX intent classification model
Optimized for speed with CPU inference
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request/Response models
class PredictionResult(BaseModel):
    intent: str = Field(description="Intent label")
    confidence: float = Field(description="Confidence score")

class ConversationClassificationRequest(BaseModel):
    messages: List[Dict[str, str]] = Field(..., description="List of messages with 'role' and 'content'")
    formatted_chat: Optional[str] = Field(None, description="Pre-formatted conversation string")
    top_k: int = Field(default=1, description="Return top K predictions", ge=1, le=5)

class ConversationClassificationResponse(BaseModel):
    original_input: Dict = Field(description="Original input data")
    intent_class: str = Field(description="Predicted intent class")
    confidence: float = Field(description="Prediction confidence")
    predictions: List[PredictionResult] = Field(description="All top-k predictions")
    processing_time_ms: float = Field(description="Processing time in milliseconds")
    model_version: str = Field(description="Model version")

class BatchConversationRequest(BaseModel):
    conversations: List[Dict] = Field(..., description="List of conversation objects", min_items=1, max_items=50)
    top_k: int = Field(default=1, description="Return top K predictions", ge=1, le=5)

class BatchConversationResponse(BaseModel):
    results: List[ConversationClassificationResponse] = Field(description="List of classification results")
    processing_time_ms: float = Field(description="Total processing time")
    model_version: str = Field(description="Model version")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float

# Global model variables
tokenizer = None
model = None
model_config = None
startup_time = None

def load_model(model_path: str):
    """Load the ONNX model and tokenizer"""
    global tokenizer, model, model_config
    
    logger.info(f"Loading model from {model_path}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load ONNX model
        model = ORTModelForSequenceClassification.from_pretrained(model_path)
        
        # Load model config
        config_path = Path(model_path) / "model_config.json"
        if config_path.exists():
            with open(config_path) as f:
                model_config = json.load(f)
        else:
            model_config = {"model_version": "unknown"}
        
        logger.info("Model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

def clean_conversation(messages: List[Dict[str, str]], formatted_chat: Optional[str] = None) -> str:
    """Clean and format conversation for model input"""
    if formatted_chat:
        # Use pre-formatted chat if provided
        text = formatted_chat
    else:
        # Format messages into conversation string
        text_parts = []
        for msg in messages:
            role = msg.get('role', '').lower()
            content = msg.get('content', '').strip()
            if content:
                if role == 'user':
                    text_parts.append(f"[USER] {content}")
                elif role == 'assistant':
                    text_parts.append(f"[ASSISTANT] {content}")
                elif role == 'system':
                    # Skip system messages or handle differently if needed
                    continue
        text = " ".join(text_parts)
    
    # Basic cleaning (you can expand this with your cleaning pipeline)
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = ' '.join(text.split())  # Normalize whitespace
    
    return text
    """Make prediction on single text"""
    if not model or not tokenizer:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Tokenize
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=model_config.get("max_length", 512),
        padding=True
    )
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
    # Get top-k predictions
    top_k_probs, top_k_indices = torch.topk(probabilities, top_k, dim=-1)
    
    # Format results
    predictions = []
    for prob, idx in zip(top_k_probs[0], top_k_indices[0]):
        label = model_config.get("label_mapping", {}).get(str(idx.item()), f"label_{idx.item()}")
        predictions.append({
            "intent": label,
            "confidence": float(prob.item())
        })
    
    return predictions

def predict_batch(texts: List[str], top_k: int = 1) -> List[List[Dict[str, float]]]:
    """Make predictions on batch of texts"""
    if not model or not tokenizer:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Tokenize all texts
    inputs = tokenizer(
        texts, 
        return_tensors="pt", 
        truncation=True, 
        max_length=model_config.get("max_length", 512),
        padding=True
    )
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Process results for each text
    results = []
    for i in range(len(texts)):
        # Get top-k for this sample
        probs = probabilities[i]
        top_k_probs, top_k_indices = torch.topk(probs, top_k)
        
        predictions = []
        for prob, idx in zip(top_k_probs, top_k_indices):
            label = model_config.get("label_mapping", {}).get(str(idx.item()), f"label_{idx.item()}")
            predictions.append({
                "intent": label,
                "confidence": float(prob.item())
            })
        
        results.append(predictions)
    
    return results

def predict_conversation(messages: List[Dict[str, str]], formatted_chat: Optional[str] = None, top_k: int = 1) -> List[Dict[str, float]]:
    """Make prediction on conversation"""
    if not model or not tokenizer:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Use formatted_chat if provided, otherwise create basic format
    if formatted_chat:
        text = formatted_chat
    else:
        # Simple message concatenation - replace with your cleaning pipeline later
        text_parts = []
        for msg in messages:
            role = msg.get('role', '').lower()
            content = msg.get('content', '').strip()
            if content and role in ['user', 'assistant']:
                text_parts.append(f"[{role.upper()}] {content}")
        text = " ".join(text_parts)
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="Empty conversation")
    
    # Tokenize
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=model_config.get("max_length", 512),
        padding=True
    )
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
    # Get top-k predictions
    top_k_probs, top_k_indices = torch.topk(probabilities, top_k, dim=-1)
    
    # Format results
    predictions = []
    for prob, idx in zip(top_k_probs[0], top_k_indices[0]):
        label = model_config.get("label_mapping", {}).get(str(idx.item()), f"label_{idx.item()}")
        predictions.append(PredictionResult(
            intent=label,
            confidence=float(prob.item())
        ))
    
    return predictions

# Create FastAPI app
app = FastAPI(
    title="Intent Classification API",
    description="Fast CPU-optimized intent classification using ONNX Runtime",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global startup_time
    startup_time = time.time()
    
    # Get model path from environment or default
    import os
    model_path = os.getenv("MODEL_PATH", "./models/onnx-intent-classifier")
    
    if not load_model(model_path):
        logger.error("Failed to load model on startup")
        raise RuntimeError("Model loading failed")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_version=model_config.get("model_version", "unknown") if model_config else "unknown",
        uptime_seconds=time.time() - startup_time if startup_time else 0
    )

@app.post("/classify/conversation", response_model=ConversationClassificationResponse)
async def classify_conversation(request: ConversationClassificationRequest):
    """Classify a conversation and return intent with original data"""
    start_time = time.perf_counter()
    
    try:
        predictions = predict_conversation(request.messages, request.formatted_chat, request.top_k)
        processing_time = (time.perf_counter() - start_time) * 1000
        
        # Get top prediction
        top_prediction = predictions[0] if predictions else PredictionResult(intent="unclear", confidence=0.0)
        
        return ConversationClassificationResponse(
            original_input={
                "messages": request.messages,
                "formatted_chat": request.formatted_chat
            },
            intent_class=top_prediction.intent,
            confidence=top_prediction.confidence,
            predictions=predictions,
            processing_time_ms=processing_time,
            model_version=model_config.get("model_version", "unknown") if model_config else "unknown"
        )
    
    except Exception as e:
        logger.error(f"Conversation classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/classify/batch", response_model=BatchConversationResponse)
async def classify_batch_conversations(request: BatchConversationRequest):
    """Classify multiple conversations in batch"""
    start_time = time.perf_counter()
    
    try:
        results = []
        for conv_data in request.conversations:
            messages = conv_data.get("messages", [])
            formatted_chat = conv_data.get("formatted_chat")
            
            predictions = predict_conversation(messages, formatted_chat, request.top_k)
            top_prediction = predictions[0] if predictions else PredictionResult(intent="unclear", confidence=0.0)
            
            result = ConversationClassificationResponse(
                original_input=conv_data,
                intent_class=top_prediction.intent,
                confidence=top_prediction.confidence,
                predictions=predictions,
                processing_time_ms=0,  # Individual timing not calculated in batch
                model_version=model_config.get("model_version", "unknown") if model_config else "unknown"
            )
            results.append(result)
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return BatchConversationResponse(
            results=results,
            processing_time_ms=processing_time,
            model_version=model_config.get("model_version", "unknown") if model_config else "unknown"
        )
    
    except Exception as e:
        logger.error(f"Batch conversation classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch classification failed: {str(e)}")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log requests for monitoring"""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start_time) * 1000
    
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.2f}ms")
    return response

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "Intent Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if not model_config:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_config": model_config,
        "num_labels": len(model_config.get("label_mapping", {})),
        "labels": list(model_config.get("label_mapping", {}).values())
    }

def main():
    """Run the FastAPI server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Intent Classification API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model-path", required=True, help="Path to ONNX model directory")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    # Set model path environment variable
    import os
    os.environ["MODEL_PATH"] = args.model_path
    
    # Run server
    uvicorn.run(
        "serve_fastapi:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
        reload=False
    )

if __name__ == "__main__":
    main()