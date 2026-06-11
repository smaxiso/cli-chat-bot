"""
Advanced Bedrock Claude API with conversation management.

This adapter wraps the advanced BedrockClient from bedrock_api project
and provides a compatible interface with the existing text-to-speech application.
"""

import sys
import os
import logging
import json
import time

# Add bedrock_api to Python path
bedrock_api_path = os.path.join(os.path.dirname(__file__), '..', 'bedrock_api')
if bedrock_api_path not in sys.path:
    sys.path.insert(0, bedrock_api_path)

try:
    from bedrock_api.services.bedrock_client import BedrockClient
    from bedrock_api.utils.logger import setup_logger
    from bedrock_api.utils.token_usage import estimate_cost
except ImportError as e:
    logging.error(f"Failed to import bedrock_api modules: {e}")
    logging.error(f"Make sure bedrock_api is in the correct location: {bedrock_api_path}")
    raise


class BedrockClaudeAdvancedAPI:
    """
    Advanced Bedrock API with conversation management and token tracking.
    
    This class provides a drop-in replacement for BedrockClaudeAPI with:
    - Automatic conversation history management
    - Token-aware history trimming
    - Cost tracking
    - Streaming support (optional)
    """
    
    def __init__(self, config, debug=False, conversation_manager=None, cache=None):
        """
        Initialize the advanced Bedrock API.
        
        Args:
            config: Configuration dictionary (compatible with existing format)
                   Expected format:
                   {
                       "aws": {
                           "region_name": "..."
                       },
                       "model_id": "..."
                   }
            debug: Enable debug logging
            conversation_manager: Optional shared ConversationManager instance
            cache: Optional ResponseCache instance for response caching
        """
        self.logger = setup_logger(__name__, debug=debug)
        self.debug = debug
        self.cache = cache
        
        # Convert config format to match bedrock_api format
        bedrock_config = self._convert_config(config)
        
        # Initialize the advanced Bedrock client
        try:
            self.client = BedrockClient(bedrock_config, debug=debug)
        except Exception as e:
            self.logger.error(f"Failed to initialize Bedrock client: {e}")
            raise
        
        # Check if we're quick-resuming with a specific model
        resume_model = config.get("_resume_model_id")
        if resume_model:
            self.client.model_id = resume_model
            print(f"✓ Resumed with model: {resume_model}")
        else:
            # Prompt user to select a model from available Bedrock models
            self._prompt_model_selection()
        
        # Use shared conversation manager if provided, otherwise use client's manager
        if conversation_manager:
            self.conversation_manager = conversation_manager
            # Replace client's conversation manager with shared one
            self.client.conversation_manager = conversation_manager
        else:
            # Access conversation manager from client
            self.conversation_manager = self.client.conversation_manager
        
        # Track session usage for display
        self.session_totals = {
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0,
            'total_cost': 0.0
        }
        
        # Default to using history (can be toggled)
        self.use_history = True
        
        self.logger.info("Advanced Bedrock API initialized with conversation management")
    
    def _prompt_model_selection(self):
        """Fetch available models from Bedrock (with caching) and let user choose."""
        valid_models = self._get_cached_or_fetch_models()
        
        if not valid_models:
            print(f"No active models found. Using default: {self.client.model_id}")
            return
        
        # Display available models
        self._display_model_list(valid_models)
        
        # Prompt selection
        self._do_model_selection(valid_models)
    
    def _get_cached_or_fetch_models(self):
        """Get models from cache or fetch from API. Cache for 3 days."""
        cache_dir = os.path.join(os.path.dirname(__file__), '..', 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, 'bedrock_models_cache.json')
        cache_ttl_seconds = 3 * 24 * 60 * 60  # 3 days
        
        # Try loading from cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                
                cached_at = cached.get('cached_at', 0)
                if (time.time() - cached_at) < cache_ttl_seconds:
                    models = cached.get('models', [])
                    if models:
                        print("\nUsing cached model list (use /refresh-models to update)")
                        return models
            except (json.JSONDecodeError, IOError):
                pass
        
        # Fetch from API
        return self._fetch_and_cache_models(cache_file)
    
    def _fetch_and_cache_models(self, cache_file=None):
        """Fetch models from Bedrock API, cache them, return processed list."""
        print("\nFetching available models from Bedrock...")
        try:
            models = self.client.list_available_models(foundation_model=True, by_provider="Anthropic")
            
            valid_models = []
            for m in models:
                model_id = m.get('modelId', '')
                model_name = m.get('modelName', model_id)
                lifecycle_status = m.get('modelLifecycle', {}).get('status', '').upper()
                inference_types = m.get('inferenceTypesSupported', [])
                
                # Skip embedding, image, video models
                if any(x in model_id.lower() for x in ['embed', 'image', 'video']):
                    continue
                
                # Skip legacy/EOL models
                if lifecycle_status in ('LEGACY', 'EOL'):
                    continue
                
                # Determine the correct invocation ID
                if 'ON_DEMAND' in inference_types:
                    invocation_id = model_id
                    note = ""
                else:
                    invocation_id = f"us.{model_id}"
                    note = " (inference profile)"
                
                valid_models.append({
                    'id': invocation_id,
                    'name': f"{model_name}{note}",
                    'raw_id': model_id
                })
            
            # Save to cache
            if valid_models and cache_file:
                try:
                    with open(cache_file, 'w') as f:
                        json.dump({
                            'cached_at': time.time(),
                            'models': valid_models
                        }, f, indent=2)
                except IOError as e:
                    self.logger.debug(f"Could not write model cache: {e}")
            
            return valid_models
            
        except Exception as e:
            print(f"Could not list models ({e}).")
            return []
    
    def _display_model_list(self, valid_models):
        """Display model list to user."""
        print("\n" + "=" * 70)
        print("Available Bedrock Models:")
        print("-" * 70)
        for i, m in enumerate(valid_models, 1):
            marker = " ← current" if m['id'] == self.client.model_id else ""
            print(f"  {i}. {m['name']}")
            print(f"     {m['id']}{marker}")
        print("=" * 70)
    
    def _do_model_selection(self, valid_models):
        """Prompt user to select from model list."""
        while True:
            choice = input("\nSelect model (number) or press Enter for current: ").strip()
            
            if not choice:
                print(f"Using: {self.client.model_id}")
                return
            
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(valid_models):
                    selected = valid_models[idx]['id']
                    self.client.model_id = selected
                    print(f"✓ Model set to: {selected}")
                    return
            
            print("Invalid selection. Enter a number from the list.")
    
    def switch_model(self):
        """Re-run model selection (called from /switch-model command)."""
        self._prompt_model_selection()
    
    def refresh_models(self):
        """Force refresh model cache and re-select."""
        cache_dir = os.path.join(os.path.dirname(__file__), '..', 'cache')
        cache_file = os.path.join(cache_dir, 'bedrock_models_cache.json')
        if os.path.exists(cache_file):
            os.remove(cache_file)
        self._prompt_model_selection()
    
    def _convert_config(self, config):
        """
        Convert text-to-speech config format to bedrock_api format.
        
        Args:
            config: Original config format
        
        Returns:
            dict: Config in bedrock_api format
        """
        bedrock_config = {
            "aws": config.get("aws", {}),
            "bedrock": {
                "model_id": config.get("model_id", "anthropic.claude-3-haiku-20240307-v1:0"),
                "default_temperature": 0.7,
                "default_top_p": 0.9,
                "default_max_tokens": 1024,
                "max_conversation_tokens": 160000,  # 80% of 200K context window
                "conversation_trim_strategy": "smart",
                "reserve_tokens": 1000
            },
            "sso": config.get("sso", {})
        }
        return bedrock_config
    
    def get_response(self, prompt, debug=False, use_history=None):
        """
        Get AI response (compatible with existing interface).
        
        Args:
            prompt: User prompt/question
            debug: Enable debug logging (overrides instance default)
            use_history: Use conversation history (default: self.use_history)
                        If None, uses instance default
        
        Returns:
            str: AI response text
        """
        if use_history is None:
            use_history = self.use_history
        
        # Get conversation ID for per-conversation caching (much faster than hashing history)
        conversation_id = getattr(self, '_conversation_id', None)
        
        # Check cache first
        if self.cache:
            cached = self.cache.get(
                query=prompt,
                conversation_id=conversation_id,  # Per-conversation caching (fast!)
                model_id=self.client.model_id,
                temperature=self.client.default_temperature,
                max_tokens=self.client.default_max_tokens
            )
            
            if cached:
                if debug or self.debug:
                    self.logger.debug("Cache hit for query")
                # Still need to add to conversation history even on cache hit
                if use_history and self.conversation_manager:
                    self.conversation_manager.add_user_message(prompt)
                    self.conversation_manager.add_assistant_message(cached)
                return cached, True  # (response, is_cached)
        
        # Cache miss - call API
        try:
            result = self.client.get_response(
                prompt=prompt,
                use_history=use_history,
                debug=debug or self.debug
            )
            
            response_text = result.get('response', '')
            
            # Store in cache (per-conversation, fast!)
            if self.cache and response_text:
                self.cache.set(
                    query=prompt,
                    response=response_text,
                    conversation_id=conversation_id,  # Per-conversation caching
                    model_id=self.client.model_id,
                    temperature=self.client.default_temperature,
                    max_tokens=self.client.default_max_tokens
                )
            
            # Track session usage
            usage = result.get('usage', {})
            input_tokens = usage.get('input_tokens', 0)
            output_tokens = usage.get('output_tokens', 0)
            
            self.session_totals['input_tokens'] += input_tokens
            self.session_totals['output_tokens'] += output_tokens
            self.session_totals['total_tokens'] += (input_tokens + output_tokens)
            
            # Calculate and track cost
            cost = estimate_cost(
                input_tokens,
                output_tokens,
                self.client.model_id
            )
            if cost:
                self.session_totals['total_cost'] += cost
            
            if self.debug:
                self.logger.debug(
                    f"Tokens - Input: {input_tokens}, Output: {output_tokens}, "
                    f"Cost: ${cost:.6f if cost else 0}"
                )
            
            return response_text, False  # (response, is_cached)
            
        except Exception as e:
            error_msg = f"Bedrock API Error: {str(e)}"
            self.logger.error(error_msg)
            return error_msg, False  # (response, is_cached)
    
    def get_response_stream(self, prompt, debug=False, use_history=None):
        """
        Get AI response with streaming (compatible with existing interface).
        
        Args:
            prompt: User prompt/question
            debug: Enable debug logging (overrides instance default)
            use_history: Use conversation history (default: self.use_history)
                        If None, uses instance default
        
        Yields:
            dict: Dictionary with 'chunk' (str) for text chunks, or 'usage' (dict) for final token usage
        """
        if use_history is None:
            use_history = self.use_history
        
        try:
            response_text = ""
            usage_info = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
            
            for event in self.client.get_response_stream(
                prompt=prompt,
                use_history=use_history,
                debug=debug or self.debug
            ):
                if 'chunk' in event:
                    response_text += event['chunk']
                    yield event
                elif 'usage' in event:
                    usage_info = event['usage']
                    yield event
            
            # Track session usage after streaming completes
            input_tokens = usage_info.get('input_tokens', 0)
            output_tokens = usage_info.get('output_tokens', 0)
            
            self.session_totals['input_tokens'] += input_tokens
            self.session_totals['output_tokens'] += output_tokens
            self.session_totals['total_tokens'] += (input_tokens + output_tokens)
            
            # Calculate and track cost
            cost = estimate_cost(
                input_tokens,
                output_tokens,
                self.client.model_id
            )
            if cost:
                self.session_totals['total_cost'] += cost
            
            if self.debug:
                self.logger.debug(
                    f"Tokens - Input: {input_tokens}, Output: {output_tokens}, "
                    f"Cost: ${cost:.6f if cost else 0}"
                )
            
        except Exception as e:
            error_msg = f"Bedrock API Error: {str(e)}"
            self.logger.error(error_msg)
            yield {'chunk': error_msg, 'usage': {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}}
    
    def clear_conversation(self):
        """
        Clear conversation history.
        
        Returns:
            int: Number of messages cleared
        """
        count = self.conversation_manager.clear()
        self.logger.info(f"Conversation history cleared ({count} messages)")
        return count
    
    def get_conversation_stats(self):
        """
        Get conversation and session statistics.
        
        Returns:
            dict: Statistics including conversation history and session totals
        """
        conv_stats = self.conversation_manager.get_stats()
        
        return {
            **conv_stats,
            'session_input_tokens': self.session_totals['input_tokens'],
            'session_output_tokens': self.session_totals['output_tokens'],
            'session_total_tokens': self.session_totals['total_tokens'],
            'session_total_cost': self.session_totals['total_cost'],
            'model_id': self.client.model_id
        }
    
    def enable_history(self):
        """Enable conversation history."""
        self.use_history = True
        self.logger.info("Conversation history enabled")
    
    def disable_history(self):
        """Disable conversation history (stateless mode)."""
        self.use_history = False
        self.logger.info("Conversation history disabled")
    
    def get_history(self):
        """Get conversation history."""
        return self.conversation_manager.get_history()
    
    def reset_session_totals(self):
        """Reset session token and cost totals."""
        self.session_totals = {
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0,
            'total_cost': 0.0
        }
        self.logger.info("Session totals reset")

