"""
AWS Session Manager with SSO support.
"""
import boto3
import subprocess
from botocore.exceptions import TokenRetrievalError, BotoCoreError, ClientError

from bedrock_api.utils.logger import setup_logger


class AWSSessionManager:
    """Manages AWS session with SSO support."""
    
    def __init__(self, profile_name, region_name, sso_script_path=None, login_timeout=300):
        """
        Initialize AWS Session Manager.
        
        Args:
            profile_name: AWS profile name to use
            region_name: AWS region name
            sso_script_path: Optional path to SSO login script
            login_timeout: Timeout for SSO login in seconds (default: 300)
        """
        self.profile_name = profile_name
        self.region_name = region_name
        self.sso_script_path = sso_script_path
        self.login_timeout = login_timeout
        self.logger = setup_logger(__name__)
    
    def get_session(self):
        """
        Get AWS session with automatic SSO fallback.
        
        Returns:
            boto3.Session: Configured AWS session
        
        Raises:
            RuntimeError: If session creation fails
        """
        try:
            session = boto3.Session(profile_name=self.profile_name)
            session.client('sts').get_caller_identity()
            self.logger.info(f"AWS session created using profile {self.profile_name}")
            return session
        except (TokenRetrievalError, BotoCoreError, ClientError) as e:
            self.logger.warning(
                f"Failed to create AWS session: {e}. Falling back to AWS SSO login."
            )
            return self.retry_with_sso_login()
        except Exception as e:
            self.logger.error(f"Unexpected error during session creation: {e}")
            try:
                return self.retry_with_sso_login()
            except Exception as fallback_error:
                raise RuntimeError(
                    f"Failed to establish AWS session. Error: {fallback_error}"
                )
    
    def retry_with_sso_login(self):
        """
        Retry creating the session after performing SSO login.
        
        Returns:
            boto3.Session: Configured AWS session
        
        Raises:
            RuntimeError: If SSO login or session creation fails
        """
        try:
            self.execute_sso_login()
            session = boto3.Session(profile_name=self.profile_name)
            session.client('sts').get_caller_identity()
            self.logger.info(
                f"AWS session created using profile {self.profile_name} after SSO login"
            )
            return session
        except (TokenRetrievalError, BotoCoreError, ClientError) as e:
            self.logger.error(f"Failed to create AWS session after SSO login: {e}")
            raise RuntimeError("AWS SSO login failed")
        except Exception as e:
            self.logger.error(f"Unexpected error during SSO retry: {e}")
            raise RuntimeError("Unexpected failure during AWS session retry")
    
    def execute_sso_login(self):
        """
        Execute SSO login using automation script or AWS CLI.
        
        Raises:
            RuntimeError: If SSO login fails or times out
        """
        try:
            self.logger.info("Executing AWS SSO login using automation script")
            
            # Try using custom SSO login script first if provided
            if self.sso_script_path:
                try:
                    subprocess.run(
                        [self.sso_script_path],
                        check=True,
                        timeout=self.login_timeout
                    )
                    self.logger.info(
                        "AWS SSO login completed successfully using automation script"
                    )
                    return
                except (subprocess.CalledProcessError, FileNotFoundError):
                    self.logger.info(
                        f"SSO script not found or failed, falling back to AWS CLI"
                    )
            
            # Fallback to standard AWS CLI SSO login
            self.logger.info("Falling back to standard AWS CLI SSO login")
            subprocess.run(
                ['aws', 'sso', 'login', '--profile', self.profile_name],
                check=True,
                timeout=self.login_timeout
            )
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

