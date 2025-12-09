"""
Bedrock API Client for Claude models.
"""
import boto3
import json
import sys
from bedrock_api.services.aws_session_manager import AWSSessionManager
from bedrock_api.services.conversation_manager import ConversationManager
from bedrock_api.utils.logger import setup_logger
from bedrock_api.utils.error_handler import is_token_expired_error, format_bedrock_error


class BedrockClient:
    """Handles requests to Claude AI API via AWS Bedrock."""
    
    def __init__(self, config, debug=False):
        """
        Initialize the Bedrock client with given configuration.
        
        Args:
            config: Configuration dictionary
            debug: Enable debug logging
        
        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If AWS session or Bedrock client creation fails
        """
        self.logger = setup_logger(__name__, debug=debug)
        self.debug = debug
        
        # Extract configuration sections
        aws_config = config.get("aws", {})
        bedrock_config = config.get("bedrock", {})
        sso_config = config.get("sso", {})
        
        # Support both SSO profile-based and direct credential configurations
        if aws_config and "profile_name" in aws_config:
            # Use SSO profile-based authentication (recommended)
            self.profile_name = aws_config.get("profile_name")
            self.region_name = aws_config.get("region_name", "us-east-1")
            self.iam_role = aws_config.get("iam_role")  # Optional for logging
            
            # Get SSO configuration
            sso_script_path = sso_config.get("sso_script_path")
            login_timeout = sso_config.get("login_timeout", 300)
            
            # Initialize AWS session using session manager
            try:
                self.logger.info(
                    f"Initializing AWS session with SSO profile: {self.profile_name}"
                )
                aws_session_manager = AWSSessionManager(
                    profile_name=self.profile_name,
                    region_name=self.region_name,
                    sso_script_path=sso_script_path,
                    login_timeout=login_timeout
                )
                self.session = aws_session_manager.get_session()
                
            except Exception as e:
                self.logger.error(f"Failed to initialize AWS session: {e}")
                raise
                
        elif "aws_access_key_id" in config:
            # Fallback to direct credential authentication
            self.aws_access_key_id = config.get("aws_access_key_id")
            self.aws_secret_access_key = config.get("aws_secret_access_key")
            self.region_name = config.get("region_name", "us-east-1")
            
            try:
                self.session = boto3.Session(
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key,
                    region_name=self.region_name
                )
                self.logger.info("Initialized AWS session with direct credentials")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize AWS session: {e}")
                raise
        else:
            raise ValueError(
                "Invalid configuration: must provide either 'aws.profile_name' "
                "or 'aws_access_key_id'"
            )
        
        # Get Bedrock configuration
        self.model_id = bedrock_config.get(
            "model_id",
            config.get("model_id", "anthropic.claude-3-sonnet-20240229-v1:0")
        )
        self.default_temperature = bedrock_config.get("default_temperature", 0.7)
        self.default_top_p = bedrock_config.get("default_top_p", 0.9)
        self.default_max_tokens = bedrock_config.get("default_max_tokens", 1024)
        
        # Initialize conversation manager for stateful conversations
        max_history = bedrock_config.get("max_conversation_history", None)
        max_tokens = bedrock_config.get("max_conversation_tokens", None)
        trim_strategy = bedrock_config.get("conversation_trim_strategy", "smart")
        reserve_tokens = bedrock_config.get("reserve_tokens", 1000)
        
        # Default max_tokens based on model context window if not specified
        # Claude 3.5 Sonnet: 200K context, use 80% = 160K tokens for history
        if max_tokens is None and "claude-3-5-sonnet" in self.model_id.lower():
            max_tokens = 160000
        elif max_tokens is None and "claude-3" in self.model_id.lower():
            max_tokens = 160000  # Claude 3 models have 200K context
        
        self._conversation_manager = ConversationManager(
            max_history=max_history,
            max_tokens=max_tokens,
            trim_strategy=trim_strategy,
            reserve_tokens=reserve_tokens
        )
        
        # Create Bedrock clients
        try:
            self.bedrock_runtime = self.session.client(
                'bedrock-runtime',
                region_name=self.region_name
            )
            self.bedrock = self.session.client(
                'bedrock',
                region_name=self.region_name
            )
            
            self.logger.info(
                f"Successfully initialized Bedrock client in region {self.region_name}"
            )
            if hasattr(self, 'profile_name'):
                self.logger.info(f"Using AWS profile: {self.profile_name}")
            if hasattr(self, 'iam_role'):
                self.logger.info(f"IAM role: {self.iam_role}")
                
        except Exception as e:
            self.logger.error(f"Failed to create Bedrock clients: {e}")
            raise
    
    def get_response(
        self,
        prompt,
        model_id=None,
        temperature=None,
        top_p=None,
        max_tokens=None,
        debug=None,
        use_history=True
    ):
        """
        Fetch AI response from Claude via AWS Bedrock.
        
        Args:
            prompt: User prompt/question
            model_id: Model ID to use (overrides config default)
            temperature: Temperature parameter (overrides config default)
            top_p: Top-p parameter (overrides config default)
            max_tokens: Max tokens parameter (overrides config default)
            debug: Enable debug logging (overrides instance default)
            use_history: If True, includes conversation history (default: True)
        
        Returns:
            dict: Dictionary with 'response' (str) and 'usage' (dict with input_tokens, output_tokens, total_tokens)
        """
        # Use provided parameters or fall back to defaults
        model = model_id or self.model_id
        temp = temperature if temperature is not None else self.default_temperature
        tp = top_p if top_p is not None else self.default_top_p
        max_tok = max_tokens if max_tokens is not None else self.default_max_tokens
        dbg = debug if debug is not None else self.debug
        
        try:
            result = self._make_bedrock_request(prompt, model, temp, tp, max_tok, dbg, use_history)
            
            # Add to conversation history if using history
            if use_history:
                self.conversation_manager.add_user_message(prompt)
                self.conversation_manager.add_assistant_message(result['response'])
            
            return result
        except Exception as e:
            # Check if this is a token expiration error
            if is_token_expired_error(e):
                self.logger.warning(
                    "SSO token expired, attempting automatic re-authentication..."
                )
                try:
                    # Re-initialize AWS session with fresh SSO login
                    self._refresh_aws_session()
                    # Retry the request with fresh credentials
                    return self._make_bedrock_request(
                        prompt, model, temp, tp, max_tok, dbg, use_history
                    )
                except Exception as retry_error:
                    error_message = (
                        f"Bedrock API Error after re-authentication: "
                        f"{str(retry_error)}"
                    )
                    self.logger.error(error_message)
                    error_response = format_bedrock_error(retry_error)
                    return {
                        'response': error_response,
                        'usage': {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
                    }
            else:
                # Handle other types of errors normally
                error_message = f"Bedrock API Error: {str(e)}"
                self.logger.error(error_message)
                error_response = format_bedrock_error(e)
                return {
                    'response': error_response,
                    'usage': {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
                }
    
    def get_response_stream(
        self,
        prompt,
        model_id=None,
        temperature=None,
        top_p=None,
        max_tokens=None,
        debug=None,
        use_history=True
    ):
        """
        Fetch AI response from Claude via AWS Bedrock with streaming.
        
        Args:
            prompt: User prompt/question
            model_id: Model ID to use (overrides config default)
            temperature: Temperature parameter (overrides config default)
            top_p: Top-p parameter (overrides config default)
            max_tokens: Max tokens parameter (overrides config default)
            debug: Enable debug logging (overrides instance default)
            use_history: If True, includes conversation history (default: True)
        
        Yields:
            dict: Dictionary with 'chunk' (str) for text chunks, or 'usage' (dict) for final token usage
        """
        # Use provided parameters or fall back to defaults
        model = model_id or self.model_id
        temp = temperature if temperature is not None else self.default_temperature
        tp = top_p if top_p is not None else self.default_top_p
        max_tok = max_tokens if max_tokens is not None else self.default_max_tokens
        dbg = debug if debug is not None else self.debug
        
        try:
            response_text = ""
            usage_info = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
            
            for event in self._make_bedrock_streaming_request(prompt, model, temp, tp, max_tok, dbg, use_history):
                if 'chunk' in event:
                    response_text += event['chunk']
                    yield event
                elif 'usage' in event:
                    usage_info = event['usage']
                    yield event
            
            # Add to conversation history if using history
            if use_history and response_text:
                self.conversation_manager.add_user_message(prompt)
                self.conversation_manager.add_assistant_message(response_text)
        except Exception as e:
            # Check if this is a token expiration error
            if is_token_expired_error(e):
                self.logger.warning(
                    "SSO token expired, attempting automatic re-authentication..."
                )
                try:
                    # Re-initialize AWS session with fresh SSO login
                    self._refresh_aws_session()
                    # Retry the request with fresh credentials
                    yield from self._make_bedrock_streaming_request(prompt, model, temp, tp, max_tok, dbg, use_history)
                except Exception as retry_error:
                    error_message = (
                        f"Bedrock API Error after re-authentication: "
                        f"{str(retry_error)}"
                    )
                    self.logger.error(error_message)
                    error_response = format_bedrock_error(retry_error)
                    yield {
                        'chunk': error_response,
                        'usage': {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
                    }
            else:
                # Handle other types of errors normally
                error_message = f"Bedrock API Error: {str(e)}"
                self.logger.error(error_message)
                error_response = format_bedrock_error(e)
                yield {
                    'chunk': error_response,
                    'usage': {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
                }
    
    def _refresh_aws_session(self):
        """Refresh the AWS session by re-authenticating."""
        self.logger.info("Refreshing AWS session due to expired token...")
        try:
            # Get SSO config for session manager
            sso_config = {}
            # Try to get from config if available
            if hasattr(self, '_config'):
                sso_config = self._config.get("sso", {})
            
            # Create a new session manager and get fresh session
            aws_session_manager = AWSSessionManager(
                profile_name=self.profile_name,
                region_name=self.region_name,
                sso_script_path=sso_config.get("sso_script_path"),
                login_timeout=sso_config.get("login_timeout", 300)
            )
            self.session = aws_session_manager.get_session()
            
            # Recreate Bedrock clients with fresh session
            self.bedrock_runtime = self.session.client(
                'bedrock-runtime',
                region_name=self.region_name
            )
            self.bedrock = self.session.client(
                'bedrock',
                region_name=self.region_name
            )
            
            self.logger.info("AWS session refreshed successfully")
        except Exception as e:
            self.logger.error(f"Failed to refresh AWS session: {e}")
            raise RuntimeError(f"Failed to refresh AWS session: {e}")
    
    def clear_conversation(self):
        """Clear the conversation history."""
        count = self.conversation_manager.clear()
        self.logger.info(f"Conversation history cleared ({count} messages removed)")
        return count
    
    def get_conversation_history(self, include_timestamps: bool = False):
        """Get the current conversation history."""
        return self.conversation_manager.get_history(include_timestamps=include_timestamps)
    
    @property
    def conversation_manager(self):
        """Get the conversation manager instance."""
        return self._conversation_manager
    
    @conversation_manager.setter
    def conversation_manager(self, value):
        """Set the conversation manager instance."""
        self._conversation_manager = value
    
    def _make_bedrock_request(
        self, prompt, model_id, temperature, top_p, max_tokens, debug, use_history=True
    ):
        """
        Make the actual request to Bedrock.
        
        Args:
            prompt: User prompt
            model_id: Model ID to use
            temperature: Temperature parameter
            top_p: Top-p parameter
            max_tokens: Max tokens parameter
            debug: Enable debug logging
            use_history: If True, includes conversation history
        
        Returns:
            dict: Dictionary with 'response' (str) and 'usage' (dict)
        """
        # Determine if this is a legacy model or modern model
        is_legacy = (
            "claude-v2" in model_id.lower() or
            "claude-instant" in model_id.lower()
        )
        
        if is_legacy:
            # Legacy Claude format - build prompt with history
            if use_history:
                history_text = self.conversation_manager.get_legacy_format(include_current_prompt=prompt)
            else:
                history_text = f"\n\nHuman: {prompt}\n\nAssistant:"
            
            body = json.dumps({
                "prompt": history_text,
                "max_tokens_to_sample": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            })
        else:
            # Modern Claude format (Claude 3 and newer)
            if use_history:
                messages = self.conversation_manager.get_messages_for_provider(
                    provider="bedrock",
                    include_current_user_message=prompt
                )
            else:
                # No history - just current message
                messages = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }]
            
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "messages": messages
            })
        
        if debug:
            self.logger.debug(f"Invoking Bedrock model: {model_id}")
            self.logger.debug(f"Request body: {body}")
        
        # Invoke the model
        response = self.bedrock_runtime.invoke_model(
            modelId=model_id,
            body=body
        )
        
        # Parse the response
        response_body = json.loads(response.get('body').read())
        
        if debug:
            self.logger.debug(f"Raw response: {response_body}")
        
        # Extract response text based on format
        if is_legacy:
            response_text = response_body.get("completion", "")
        else:
            response_text = ""
            for item in response_body.get('content', []):
                if item.get('type') == 'text':
                    response_text += item.get('text', '')
        
        # Extract token usage
        usage_info = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
        if 'usage' in response_body:
            usage = response_body['usage']
            input_tokens = usage.get('input_tokens', 0)
            output_tokens = usage.get('output_tokens', 0)
            usage_info = {
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens
            }
        
        if debug:
            self.logger.debug(
                f"Token usage - Input: {usage_info['input_tokens']}, "
                f"Output: {usage_info['output_tokens']}, "
                f"Total: {usage_info['total_tokens']}"
            )
        
        return {
            'response': response_text,
            'usage': usage_info
        }
    
    def _make_bedrock_streaming_request(
        self, prompt, model_id, temperature, top_p, max_tokens, debug, use_history=True
    ):
        """
        Make a streaming request to Bedrock.
        
        Args:
            prompt: User prompt
            model_id: Model ID to use
            temperature: Temperature parameter
            top_p: Top-p parameter
            max_tokens: Max tokens parameter
            debug: Enable debug logging
            use_history: If True, includes conversation history
        
        Yields:
            dict: Dictionary with 'chunk' (str) for text chunks, or 'usage' (dict) for final token usage
        """
        # Determine if this is a legacy model or modern model
        is_legacy = (
            "claude-v2" in model_id.lower() or
            "claude-instant" in model_id.lower()
        )
        
        if is_legacy:
            # Legacy Claude format - build prompt with history
            if use_history:
                history_text = self.conversation_manager.get_legacy_format(include_current_prompt=prompt)
            else:
                history_text = f"\n\nHuman: {prompt}\n\nAssistant:"
            
            body = json.dumps({
                "prompt": history_text,
                "max_tokens_to_sample": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            })
        else:
            # Modern Claude format (Claude 3 and newer)
            if use_history:
                messages = self.conversation_manager.get_messages_for_provider(
                    provider="bedrock",
                    include_current_user_message=prompt
                )
            else:
                # No history - just current message
                messages = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }]
            
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "messages": messages
            })
        
        if debug:
            self.logger.debug(f"Invoking Bedrock model with streaming: {model_id}")
            self.logger.debug(f"Request body: {body}")
        
        # Invoke the model with streaming
        response = self.bedrock_runtime.invoke_model_with_response_stream(
            modelId=model_id,
            body=body
        )
        
        # Process the event stream
        response_text = ""
        usage_info = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
        
        stream = response.get('body')
        if stream:
            for event in stream:
                if debug:
                    self.logger.debug(f"Raw event keys: {event.keys() if hasattr(event, 'keys') else 'no keys'}")
                
                if 'chunk' in event:
                    chunk = event.get('chunk')
                    if chunk:
                        # Handle bytes - decode if needed
                        chunk_bytes = chunk.get('bytes')
                        if chunk_bytes:
                            # If it's already bytes, decode it; if it's a file-like object, read it
                            if isinstance(chunk_bytes, bytes):
                                chunk_str = chunk_bytes.decode('utf-8')
                            else:
                                chunk_str = chunk_bytes.read().decode('utf-8')
                            
                            try:
                                chunk_data = json.loads(chunk_str)
                            except json.JSONDecodeError as e:
                                if debug:
                                    self.logger.debug(f"Failed to parse chunk JSON: {e}, chunk_str: {chunk_str[:100]}")
                                continue
                            
                            if debug:
                                self.logger.debug(f"Stream chunk data keys: {list(chunk_data.keys())}")
                                self.logger.debug(f"Stream chunk: {chunk_data}")
                            
                            # Handle different event types by checking 'type' field first
                            event_type = chunk_data.get('type')
                            
                            if event_type == 'message_start':
                                # Message start event - contains input tokens in message.usage
                                message = chunk_data.get('message', {})
                                if 'usage' in message:
                                    usage = message.get('usage', {})
                                    input_tokens = usage.get('input_tokens', 0)
                                    # Store input tokens (output tokens here are initial, not final)
                                    usage_info['input_tokens'] = input_tokens
                                    if debug:
                                        self.logger.debug(f"Input tokens from message_start: {input_tokens}")
                            
                            elif event_type == 'message_delta':
                                # Message delta event - contains final output tokens
                                if 'usage' in chunk_data:
                                    usage = chunk_data.get('usage', {})
                                    output_tokens = usage.get('output_tokens', 0)
                                    usage_info['output_tokens'] = output_tokens
                                    usage_info['total_tokens'] = usage_info['input_tokens'] + output_tokens
                                    if debug:
                                        self.logger.debug(f"Output tokens from message_delta: {output_tokens}, Total: {usage_info['total_tokens']}")
                                    # Yield usage when we have both input and output
                                    yield {'usage': usage_info}
                            
                            elif event_type == 'message_stop':
                                # Message stop event - contains final metrics in amazon-bedrock-invocationMetrics
                                if 'amazon-bedrock-invocationMetrics' in chunk_data:
                                    metrics = chunk_data.get('amazon-bedrock-invocationMetrics', {})
                                    input_tokens = metrics.get('inputTokenCount', 0)
                                    output_tokens = metrics.get('outputTokenCount', 0)
                                    if input_tokens > 0 or output_tokens > 0:
                                        usage_info = {
                                            'input_tokens': input_tokens,
                                            'output_tokens': output_tokens,
                                            'total_tokens': input_tokens + output_tokens
                                        }
                                        if debug:
                                            self.logger.debug(f"Token usage from message_stop metrics: {usage_info}")
                                        yield {'usage': usage_info}
                            
                            elif event_type == 'content_block_delta':
                                # Content block delta - contains text chunks
                                delta = chunk_data.get('delta', {})
                                if delta.get('type') == 'text_delta' and 'text' in delta:
                                    text_chunk = delta['text']
                                    response_text += text_chunk
                                    yield {'chunk': text_chunk}
                            
                            # Legacy format handling (for older models)
                            elif 'delta' in chunk_data:
                                # Text delta (content chunk)
                                delta = chunk_data.get('delta', {})
                                if 'text' in delta:
                                    text_chunk = delta['text']
                                    response_text += text_chunk
                                    yield {'chunk': text_chunk}
                            
                            elif chunk_data.get('type') == 'content_block_delta':
                                # Content block delta (modern format)
                                delta = chunk_data.get('delta', {})
                                if delta.get('type') == 'text_delta' and 'text' in delta:
                                    text_chunk = delta['text']
                                    response_text += text_chunk
                                    yield {'chunk': text_chunk}
        
        # Yield final usage information (even if 0, so caller knows we completed)
        yield {'usage': usage_info}
    
    def list_available_models(self, foundation_model=True, by_provider=None):
        """
        List available Bedrock foundation models.
        
        Args:
            foundation_model: If True, only return foundation models (default: True)
            by_provider: Filter by provider (e.g., 'Anthropic', 'Amazon', 'AI21 Labs')
        
        Returns:
            list: List of model summaries with modelId, modelName, providerName
        """
        try:
            if foundation_model:
                response = self.bedrock.list_foundation_models()
            else:
                response = self.bedrock.list_models()
            
            models = response.get('modelSummaries', [])
            
            # Filter by provider if specified
            if by_provider:
                models = [
                    m for m in models
                    if m.get('providerName', '').lower() == by_provider.lower()
                ]
            
            return models
            
        except Exception as e:
            self.logger.error(f"Error listing models: {e}")
            raise
    
    def list_gen_ai_models(self, provider=None, on_demand_only=True):
        """
        List available Gen AI models (text generation models).
        
        Args:
            provider: Filter by provider (e.g., 'Anthropic', 'Amazon', 'AI21 Labs', 'Meta')
            on_demand_only: If True, only return models that support on-demand inference
        
        Returns:
            list: List of Gen AI model summaries
        """
        try:
            models = self.list_available_models(foundation_model=True, by_provider=provider)
            
            # Filter for text generation models (exclude embedding, image, etc.)
            gen_ai_models = []
            for model in models:
                model_id = model.get('modelId', '').lower()
                
                # Exclude embedding and image models
                if any(exclude in model_id for exclude in ['embed', 'image', 'rerank']):
                    continue
                
                # Include common Gen AI model prefixes
                if any(prefix in model_id for prefix in [
                    'claude', 'titan', 'jurassic', 'llama', 'mistral', 'cohere'
                ]):
                    # Check if model supports on-demand inference
                    if on_demand_only:
                        inference_types = model.get('inferenceTypesSupported', [])
                        # If inferenceTypesSupported is empty or contains 'ON_DEMAND', include it
                        # Some models may not have this field, so we'll include them but mark them
                        if not inference_types or 'ON_DEMAND' in inference_types:
                            gen_ai_models.append(model)
                        # Exclude known inference-profile-only models
                        elif any(profile_only in model_id for profile_only in [
                            'claude-opus-4-5', 'claude-opus-4-1', 'claude-sonnet-4-5'
                        ]):
                            continue
                        else:
                            # If it has inference types but no ON_DEMAND, exclude it
                            continue
                    else:
                        gen_ai_models.append(model)
            
            return gen_ai_models
            
        except Exception as e:
            self.logger.error(f"Error listing Gen AI models: {e}")
            raise
    
    def format_models_list(self, models, show_on_demand=True):
        """
        Format model list for display.
        
        Args:
            models: List of model dictionaries
            show_on_demand: If True, add note about on-demand support
        
        Returns:
            str: Formatted string representation
        """
        if not models:
            return "No models found."
        
        lines = []
        lines.append(f"\n{'=' * 80}")
        lines.append(f"{'Available Gen AI Models (On-Demand)':^80}")
        lines.append(f"{'=' * 80}")
        lines.append(f"{'Model ID':<50} {'Provider':<20} {'Status':<10}")
        lines.append(f"{'-' * 80}")
        
        for model in models:
            model_id = model.get('modelId', 'N/A')
            provider = model.get('providerName', 'N/A')
            status = model.get('modelLifecycle', {}).get('status', 'N/A')
            lines.append(f"{model_id:<50} {provider:<20} {status:<10}")
        
        lines.append(f"{'=' * 80}")
        if show_on_demand:
            lines.append("Note: Only models supporting on-demand inference are shown.")
            lines.append("      Models requiring inference profiles are excluded.")
        lines.append("")
        
        return "\n".join(lines)

