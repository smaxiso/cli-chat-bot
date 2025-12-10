"""
Advanced Bedrock Claude API with conversation management.

This adapter wraps the advanced BedrockClient from bedrock_api project
and provides a compatible interface with the existing text-to-speech application.
"""

import sys
import os
import logging

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
                           "profile_name": "...",
                           "region_name": "...",
                           "iam_role": "..."
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
                "model_id": config.get("model_id", "anthropic.claude-3-5-sonnet-20240620-v1:0"),
                "default_temperature": 0.7,
                "default_top_p": 0.9,
                "default_max_tokens": 1024,
                "max_conversation_tokens": 160000,  # 80% of 200K context window
                "conversation_trim_strategy": "smart",
                "reserve_tokens": 1000
            },
            "sso": {
                "sso_script_path": "/Users/sumit.kumar/Documents/utils/automations/aws_sso_login.sh",
                "login_timeout": 300
            }
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

