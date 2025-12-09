#!/bin/bash

# 1. Get the current dynamic Host IP
HOST_IP=$(ip route show | grep default | awk '{print $3}')
NEW_URL="http://$HOST_IP:11434"

# 2. Ensure .env file exists
touch .env

# 3. Check if OLLAMA_BASE_URL is already in the file
if grep -q "^OLLAMA_BASE_URL=" .env; then
    # If it exists, replace it using your sed command
    sed -i "s|^OLLAMA_BASE_URL=.*|OLLAMA_BASE_URL=$NEW_URL|" .env
else
    # If it doesn't exist, append it to the end
    echo "OLLAMA_BASE_URL=$NEW_URL" >> .env
fi

# 4. Also ensure the Model is set (optional)
if ! grep -q "^OLLAMA_MODEL=" .env; then
    echo "OLLAMA_MODEL=llama3.1:latest" >> .env
fi

echo "🔄 Auto-updated .env with Host IP: $HOST_IP"