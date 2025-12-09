"""
Error handling utilities for Bedrock API.
"""


def is_token_expired_error(error):
    """
    Check if the error is related to expired SSO token.
    
    Args:
        error: Exception object or error string
    
    Returns:
        bool: True if error indicates token expiration
    """
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


def format_bedrock_error(error):
    """
    Format error message with helpful suggestions for common Bedrock errors.
    
    Args:
        error: Exception object or error string
    
    Returns:
        str: Formatted error message with helpful suggestions
    """
    error_message = f"Bedrock API Error: {str(error)}"
    error_str = str(error)
    
    # Check for common Bedrock errors and provide helpful messages
    if "ValidationException" in error_str and "on-demand throughput isn't supported" in error_str:
        error_message += (
            "\n\n⚠️  This model requires an inference profile and doesn't support on-demand inference."
            "\n\nTo see only on-demand models, use: python main.py --list-models"
            "\nOr in interactive mode, type: models"
            "\n\nSuggested on-demand alternatives:"
            "\n  - anthropic.claude-3-haiku-20240307-v1:0 (fast, cost-effective)"
            "\n  - anthropic.claude-3-5-haiku-20241022-v1:0 (improved haiku)"
            "\n  - anthropic.claude-3-7-sonnet-20250219-v1:0 (latest sonnet)"
            "\n  - anthropic.claude-sonnet-4-20250514-v1:0 (Claude 4 sonnet)"
        )
    elif "AccessDeniedException" in error_str:
        error_message += (
            "\n\nAccess denied. Please check your AWS credentials and IAM permissions for Bedrock."
        )
    elif "ThrottlingException" in error_str:
        error_message += "\n\nAPI rate limit exceeded. Please try again in a moment."
    elif "ModelNotReadyException" in error_str:
        error_message += "\n\nThe model is currently not ready. Please try again later."
    elif "NoCredentialsError" in error_str:
        error_message += "\n\nAWS credentials not found. Please check your configuration."
    
    return error_message

