import boto3
import json
import logging
import subprocess
from botocore.exceptions import TokenRetrievalError, BotoCoreError, ClientError

class AWSSessionManager:
    """Manages AWS session with SSO support"""
    
    def __init__(self, profile_name, region_name):
        self.profile_name = profile_name
        self.region_name = region_name
        self.logger = logging.getLogger(__name__)

    def get_session(self):
        """Get AWS session with automatic SSO fallback"""
        try:
            session = boto3.Session(profile_name=self.profile_name)
            session.client('sts').get_caller_identity()
            self.logger.info(f"AWS session created using profile {self.profile_name}")
            return session
        except (TokenRetrievalError, BotoCoreError, ClientError) as e:
            self.logger.warning(f"Failed to create AWS session: {e}. Falling back to AWS SSO login.")
            return self.retry_with_sso_login()
        except Exception as e:
            self.logger.error(f"Unexpected error during session creation: {e}")
            try:
                return self.retry_with_sso_login()
            except Exception as fallback_error:
                raise RuntimeError(f"Failed to establish AWS session. Error: {fallback_error}")

    def retry_with_sso_login(self):
        """Retry creating the session after performing SSO login"""
        try:
            self.execute_sso_login()
            session = boto3.Session(profile_name=self.profile_name)
            session.client('sts').get_caller_identity()
            self.logger.info(f"AWS session created using profile {self.profile_name} after SSO login")
            return session
        except (TokenRetrievalError, BotoCoreError, ClientError) as e:
            self.logger.error(f"Failed to create AWS session after SSO login: {e}")
            raise RuntimeError("AWS SSO login failed")
        except Exception as e:
            self.logger.error(f"Unexpected error during SSO retry: {e}")
            raise RuntimeError("Unexpected failure during AWS session retry")

    def execute_sso_login(self):
        """Execute SSO login using the user's automation script"""
        try:
            self.logger.info("Executing AWS SSO login using automation script")
            # Try using the user's existing SSO login script first
            sso_script_path = "/Users/sumit.kumar/Documents/utils/automations/aws_sso_login.sh"
            try:
                subprocess.run([sso_script_path], check=True, timeout=300)
                self.logger.info("AWS SSO login completed successfully using automation script")
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Fallback to standard AWS CLI SSO login
                self.logger.info("Falling back to standard AWS CLI SSO login")
                subprocess.run(['aws', 'sso', 'login', '--profile', self.profile_name], 
                             check=True, timeout=300)
                self.logger.info("AWS SSO login completed successfully")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error executing AWS SSO login: {e}")
            raise RuntimeError("AWS SSO login failed")
        except subprocess.TimeoutExpired as e:
            self.logger.error(f"AWS SSO login timed out: {e}")
            raise RuntimeError("AWS SSO login timed out")
        except Exception as e:
            self.logger.error(f"Unexpected error during SSO login: {e}")
            raise RuntimeError("Unexpected failure during SSO login")


class BedrockClaudeAPI:
    """Handles requests to Claude AI API via AWS Bedrock."""

    def __init__(self, config, conversation_manager=None):
        """Initialize the API with given configuration.
        
        Args:
            config: Configuration dictionary
            conversation_manager: Optional shared ConversationManager instance
        """
        self.logger = logging.getLogger(__name__)
        
        # Use shared conversation manager if provided, otherwise create simple list
        self.conversation_manager = conversation_manager
        if not self.conversation_manager:
            # Fallback to simple list for backward compatibility
            self.conversation_history = []
        else:
            # Use conversation manager's history
            self.conversation_history = self.conversation_manager.conversation_history
        
        # Extract AWS configuration
        aws_config = config.get("aws", {})
        
        # Support both SSO profile-based and direct credential configurations
        if aws_config and "profile_name" in aws_config:
            # Use SSO profile-based authentication (recommended)
            self.profile_name = aws_config.get("profile_name")
            self.region_name = aws_config.get("region_name", "us-east-1")
            self.iam_role = aws_config.get("iam_role")  # Optional for logging
            self.model_id = config.get("model_id", "anthropic.claude-3-sonnet-20240229-v1:0")
            
            # Initialize AWS session using session manager
            try:
                self.logger.info(f"Initializing AWS session with SSO profile: {self.profile_name}")
                aws_session_manager = AWSSessionManager(self.profile_name, self.region_name)
                self.session = aws_session_manager.get_session()
                
            except Exception as e:
                self.logger.error(f"Failed to initialize AWS session: {e}")
                raise
                
        elif "aws_access_key_id" in config:
            # Fallback to direct credential authentication
            self.aws_access_key_id = config.get("aws_access_key_id")
            self.aws_secret_access_key = config.get("aws_secret_access_key")
            self.region_name = config.get("region_name", "us-east-1")
            self.model_id = config.get("model_id", "anthropic.claude-3-sonnet-20240229-v1:0")
            
            try:
                self.session = boto3.Session(
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key,
                    region_name=self.region_name
                )
                self.logger.info("Initialized AWS session with direct credentials")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize Bedrock client: {e}")
                raise
        else:
            raise ValueError("Invalid configuration: must provide either 'aws.profile_name' or 'aws_access_key_id'")
        
        # Create Bedrock clients
        try:
            self.bedrock_runtime = self.session.client('bedrock-runtime', region_name=self.region_name)
            self.bedrock = self.session.client('bedrock', region_name=self.region_name)
            
            self.logger.info(f"Successfully initialized Bedrock Claude API in region {self.region_name}")
            if hasattr(self, 'profile_name'):
                self.logger.info(f"Using AWS profile: {self.profile_name}")
            if hasattr(self, 'iam_role'):
                self.logger.info(f"IAM role: {self.iam_role}")
                
        except Exception as e:
            self.logger.error(f"Failed to create Bedrock clients: {e}")
            raise

    def get_response(self, prompt, debug=False):
        """Fetch AI response from Claude via AWS Bedrock."""
        try:
            return self._make_bedrock_request(prompt, debug), False
        except Exception as e:
            # Check if this is a token expiration error
            if self._is_token_expired_error(e):
                self.logger.warning("SSO token expired, attempting automatic re-authentication...")
                try:
                    # Re-initialize AWS session with fresh SSO login
                    self._refresh_aws_session()
                    # Retry the request with fresh credentials
                    return self._make_bedrock_request(prompt, debug), False
                except Exception as retry_error:
                    error_message = f"Bedrock Claude API Error after re-authentication: {str(retry_error)}"
                    self.logger.error(error_message)
                    return self._format_error_message(retry_error), False
            else:
                # Handle other types of errors normally
                error_message = f"Bedrock Claude API Error: {str(e)}"
                self.logger.error(error_message)
                return self._format_error_message(e), False

    def _is_token_expired_error(self, error):
        """Check if the error is related to expired SSO token."""
        error_str = str(error).lower()
        token_expired_indicators = [
            "token has expired",
            "tokenerror",
            "tokenretrievalerror",
            "invalidgrantexception", 
            "refresh failed",
            "sso token",
            "expired credentials"
        ]
        return any(indicator in error_str for indicator in token_expired_indicators)

    def _refresh_aws_session(self):
        """Refresh the AWS session by re-authenticating."""
        self.logger.info("Refreshing AWS session due to expired token...")
        try:
            # Create a new session manager and get fresh session
            aws_session_manager = AWSSessionManager(self.profile_name, self.region_name)
            self.session = aws_session_manager.get_session()
            
            # Recreate Bedrock clients with fresh session
            self.bedrock_runtime = self.session.client('bedrock-runtime', region_name=self.region_name)
            self.bedrock = self.session.client('bedrock', region_name=self.region_name)
            
            self.logger.info("AWS session refreshed successfully")
        except Exception as e:
            self.logger.error(f"Failed to refresh AWS session: {e}")
            raise RuntimeError(f"Failed to refresh AWS session: {e}")

    def _make_bedrock_request(self, prompt, debug=False):
        """Make the actual request to Bedrock."""
        # Get conversation history (from manager or simple list)
        if self.conversation_manager:
            history = self.conversation_manager.conversation_history
        else:
            history = self.conversation_history if hasattr(self, 'conversation_history') else []
        
        # Determine if this is a legacy model or modern model
        is_legacy = "claude-v2" in self.model_id.lower() or "claude-instant" in self.model_id.lower()
        
        if is_legacy:
            # Legacy Claude format - build conversation history
            conversation_text = ""
            for msg in history:
                role = "Human" if msg["role"] == "user" else "Assistant"
                conversation_text += f"\n\n{role}: {msg['content']}"
            conversation_text += f"\n\nHuman: {prompt}\n\nAssistant:"
            
            body = json.dumps({
                "prompt": conversation_text,
                "max_tokens_to_sample": 1024,
                "temperature": 0.7,
                "top_p": 0.9
            })
        else:
            # Modern Claude format (Claude 3 and newer) - use conversation history
            messages = []
            # Add conversation history
            for msg in history:
                messages.append({
                    "role": msg["role"],
                    "content": [{"type": "text", "text": msg["content"]}]
                })
            # Add current prompt
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            })
            
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "temperature": 0.7,
                "top_p": 0.9,
                "messages": messages
            })

        if debug:
            logging.info(f"Invoking Bedrock model: {self.model_id}")
            logging.info(f"Request body: {body}")

        # Invoke the model
        response = self.bedrock_runtime.invoke_model(
            modelId=self.model_id,
            body=body
        )

        # Parse the response
        response_body = json.loads(response.get('body').read())
        
        if debug:
            logging.info(f"Raw response: {response_body}")

        # Extract response text based on format
        if is_legacy:
            response_text = response_body.get("completion", "")
        else:
            response_text = ""
            for item in response_body.get('content', []):
                if item.get('type') == 'text':
                    response_text += item.get('text', '')

        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append({"role": "assistant", "content": response_text})
        
        # Limit conversation history to last 20 messages (10 exchanges) to prevent context overflow
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        if debug:
            # Log token usage if available
            if 'usage' in response_body:
                usage = response_body['usage']
                logging.info(f"Token usage - Input: {usage.get('input_tokens', 0)}, Output: {usage.get('output_tokens', 0)}")

        return response_text

    def _format_error_message(self, error):
        """Format error message with helpful suggestions."""
        error_message = f"Bedrock Claude API Error: {str(error)}"
        
        # Check for common Bedrock errors and provide helpful messages
        if "ValidationException" in str(error) and "on-demand throughput isn't supported" in str(error):
            error_message += "\n\nThis model requires provisioned throughput. Try using a different model like 'anthropic.claude-3-sonnet-20240229-v1:0' or 'anthropic.claude-3-haiku-20240307-v1:0'."
        elif "AccessDeniedException" in str(error):
            error_message += "\n\nAccess denied. Please check your AWS credentials and IAM permissions for Bedrock."
        elif "ThrottlingException" in str(error):
            error_message += "\n\nAPI rate limit exceeded. Please try again in a moment."
        elif "ModelNotReadyException" in str(error):
            error_message += "\n\nThe model is currently not ready. Please try again later."
        elif "NoCredentialsError" in str(error):
            error_message += "\n\nAWS credentials not found. Please check your configuration."
        
        return error_message
