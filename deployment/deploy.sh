#!/bin/bash
# Intent Classification API Deployment Script
set -e # Exit on any error

echo "🚀 Deploying Intent Classification API with Docker..."

# Detect docker compose command
if command -v docker-compose > /dev/null 2>&1; then
    DOCKER_COMPOSE="docker-compose"
elif docker compose version > /dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
else
    echo "❌ Error: Neither 'docker-compose' nor 'docker compose' found"
    exit 1
fi

echo "Using: $DOCKER_COMPOSE"

# Check if we're in the right directory
if [ ! -f "requirements.txt" ] || [ ! -d "src" ] || [ ! -d "models" ]; then
    echo "❌ Error: Please run this script from the topic-classifier root directory"
    exit 1
fi

# Check if ONNX model exists
if [ ! -d "models/onnx-intent-classifier" ]; then
    echo "❌ Error: ONNX model not found at models/onnx-intent-classifier"
    exit 1
fi

cd deployment

echo "🔨 Building Docker image..."
$DOCKER_COMPOSE build

echo "🏃 Starting API service..."
$DOCKER_COMPOSE up -d

echo "⏳ Waiting for API to be ready..."
timeout=60
counter=0

while ! curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 2
    counter=$((counter + 2))
    if [ $counter -ge $timeout ]; then
        echo "❌ API failed to start within ${timeout} seconds"
        $DOCKER_COMPOSE logs intent-classifier
        exit 1
    fi
    echo "  ... waiting ($counter/${timeout}s)"
done

echo "✅ API is ready!"
curl -s http://localhost:8000/health | python3 -m json.tool 2>/dev/null || echo "Health check successful"

echo ""
echo "🧪 Test: docker compose logs -f intent-classifier"
echo "🛑 Stop: docker compose down"