#!/bin/bash
set -e

MODELS=${OLLAMA_MODELS:-qwen3,mistral}

echo "Waiting for Ollama service to be ready..."
max_attempts=180
attempt=0

while [ $attempt -lt $max_attempts ]; do
  if curl -s http://ollama:11434/api/tags > /dev/null 2>&1; then
    echo "✓ Ollama service is ready!"
    break
  fi
  attempt=$((attempt + 1))
  echo "Attempting to connect to Ollama... ($attempt/$max_attempts)"
  sleep 2
done

if [ $attempt -eq $max_attempts ]; then
  echo "✗ Timeout waiting for Ollama service"
  exit 1
fi

# Parse and pull each model
echo "Pulling Ollama models (this may take a few minutes on first run)..."
IFS=',' read -ra MODELS_ARRAY <<< "$MODELS"
for MODEL in "${MODELS_ARRAY[@]}"; do
  MODEL=$(echo "$MODEL" | xargs)  # Trim whitespace
  echo "Pulling model '$MODEL'..."
  curl -X POST http://ollama:11434/api/pull -d "{\"name\":\"$MODEL\"}" --no-buffer 2>/dev/null || {
    echo "Warning: Could not pull model $MODEL. The model may already be pulled or there was an error."
  }
done

echo ""
echo "Verifying model availability..."
available_models=$(curl -s http://ollama:11434/api/tags | grep -o '"name":"[^"]*"' | cut -d'"' -f4 | tr '\n' ' ')
echo "Available models: $available_models"

echo ""
echo "✓ Setup complete! Your Python environment is ready."
echo "To run the sample: python sample.py"
