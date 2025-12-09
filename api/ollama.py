"""
Ollama API Integration

Supports local Ollama models with streaming and conversation management.
Prioritizes configuration from .env file for WSL compatibility.
"""

import requests
import json
import logging
import os
from typing import Optional, Iterator, Dict


class OllamaAPI:
    """Handles requests to local Ollama API."""

    def __init__(self, config, conversation_manager=None, debug=False):
        """
        Initialize the Ollama API.
        
        Args:
            config: Configuration dictionary
            conversation_manager: Optional shared ConversationManager instance
            debug: Enable debug logging
        """
        self.logger = logging.getLogger(__name__)
        self.debug = debug
        
        # 1. Load configuration (Priority: .env > config.json > defaults)
        env_vars = self._read_env_file()
        
        # URL Logic
        self.base_url = env_vars.get("OLLAMA_BASE_URL")
        if self.base_url:
            if self.debug:
                self.logger.info(f"Loaded Ollama URL from .env: {self.base_url}")
        else:
            self.base_url = config.get("base_url", "http://localhost:11434")

        # Model Logic
        self.model = env_vars.get("OLLAMA_MODEL")
        if self.model:
            if self.debug:
                self.logger.info(f"Loaded Ollama Model from .env: {self.model}")
        else:
            self.model = config.get("model", "qwen2.5-coder:14b")
        
        # Use shared conversation manager if provided
        self.conversation_manager = conversation_manager
        
        # Test connection
        try:
            # Short timeout for connection test
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                self.logger.info(f"Connected to Ollama at {self.base_url}")
            else:
                self.logger.warning(f"Ollama connection test returned status {response.status_code}")
        except Exception as e:
            self.logger.warning(f"Could not connect to Ollama at {self.base_url}: {e}")
            self.logger.warning("If on WSL, ensure OLLAMA_BASE_URL is set in .env or config points to Host IP.")
        
        self.logger.info(f"Ollama API initialized with model: {self.model}")

    def _read_env_file(self) -> Dict[str, str]:
        """Helper to read variables from .env file manually."""
        env_vars = {}
        try:
            if os.path.exists(".env"):
                with open(".env", "r") as f:
                    for line in f:
                        line = line.strip()
                        # Ignore comments and empty lines
                        if not line or line.startswith('#'):
                            continue
                        # Parse KEY=VALUE
                        if '=' in line:
                            key, value = line.split('=', 1)
                            env_vars[key.strip()] = value.strip()
        except Exception as e:
            if self.debug:
                self.logger.warning(f"Failed to read .env file: {e}")
        return env_vars

    def get_response(self, prompt, debug=False):
        """
        Fetch AI response from Ollama (non-streaming).
        
        Args:
            prompt: User prompt/question
            debug: Enable debug logging for this call
        
        Returns:
            str: AI response text
        """
        try:
            # Build messages from conversation history if available
            messages = []
            
            if self.conversation_manager:
                # Get conversation history from shared manager
                history = self.conversation_manager.get_history()
                for msg in history:
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # Add current prompt
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            # Make request to Ollama
            url = f"{self.base_url}/api/chat"
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False
            }
            
            if debug or self.debug:
                self.logger.debug(f"Sending request to Ollama: {payload}")
            
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            
            result = response.json()
            response_text = result.get("message", {}).get("content", "")
            
            # Add to conversation history if using manager
            if self.conversation_manager:
                self.conversation_manager.add_user_message(prompt)
                self.conversation_manager.add_assistant_message(response_text)
            
            if debug or self.debug:
                self.logger.debug(f"Ollama response received: {len(response_text)} characters")
            
            return response_text
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Ollama API Error: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Ollama API Error: {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    def get_response_stream(self, prompt, debug=False):
        """
        Fetch AI response from Ollama with streaming.
        
        Args:
            prompt: User prompt/question
            debug: Enable debug logging for this call
        
        Yields:
            dict: Dictionary with 'chunk' (str) for text chunks, or 'usage' (dict) for final token usage
        """
        try:
            # Build messages from conversation history if available
            messages = []
            
            if self.conversation_manager:
                # Get conversation history from shared manager
                history = self.conversation_manager.get_history()
                for msg in history:
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # Add current prompt
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            # Make streaming request to Ollama
            url = f"{self.base_url}/api/chat"
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": True
            }
            
            if debug or self.debug:
                self.logger.debug(f"Sending streaming request to Ollama: {payload}")
            
            response = requests.post(url, json=payload, stream=True, timeout=300)
            response.raise_for_status()
            
            response_text = ""
            
            # Stream the response
            for line in response.iter_lines():
                if line:
                    try:
                        chunk_data = json.loads(line)
                        
                        # Extract content from chunk
                        if "message" in chunk_data:
                            content = chunk_data["message"].get("content", "")
                            if content:
                                response_text += content
                                yield {"chunk": content}
                        
                        # Check if this is the final chunk
                        if chunk_data.get("done", False):
                            # Add to conversation history if using manager
                            if self.conversation_manager:
                                self.conversation_manager.add_user_message(prompt)
                                self.conversation_manager.add_assistant_message(response_text)
                            
                            # Yield usage information if available
                            usage = chunk_data.get("eval_count", 0)  # Ollama doesn't provide detailed usage
                            yield {
                                "usage": {
                                    "input_tokens": 0,  # Ollama doesn't provide this
                                    "output_tokens": usage,
                                    "total_tokens": usage
                                }
                            }
                            break
                            
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        if debug or self.debug:
                            self.logger.debug(f"Error processing chunk: {e}")
                        continue
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Ollama API Error: {str(e)}"
            self.logger.error(error_msg)
            yield {"chunk": error_msg, "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}
        except Exception as e:
            error_msg = f"Ollama API Error: {str(e)}"
            self.logger.error(error_msg)
            yield {"chunk": error_msg, "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}