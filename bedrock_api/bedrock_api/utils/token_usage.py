"""
Token usage formatting and cost estimation utilities.
"""

# Approximate pricing per 1M tokens (as of 2024, may vary by region)
# These are rough estimates - actual pricing may differ
PRICING = {
    'anthropic': {
        'claude-3-haiku': {'input': 0.25, 'output': 1.25},  # per 1M tokens
        'claude-3-sonnet': {'input': 3.00, 'output': 15.00},
        'claude-3-5-sonnet': {'input': 3.00, 'output': 15.00},
        'claude-3-opus': {'input': 15.00, 'output': 75.00},
    },
    'mistral': {
        'default': {'input': 0.15, 'output': 0.15},
    },
    'meta': {
        'llama3': {'input': 0.20, 'output': 0.20},
    },
    'amazon': {
        'titan': {'input': 0.80, 'output': 0.80},
    },
    'cohere': {
        'command-r': {'input': 0.50, 'output': 1.50},
    }
}


def format_token_usage(usage, model_id=None):
    """
    Format token usage information with optional cost estimation.
    
    Args:
        usage: Dictionary with 'input_tokens', 'output_tokens', 'total_tokens'
        model_id: Optional model ID for cost estimation
    
    Returns:
        str: Formatted token usage string
    """
    input_tokens = usage.get('input_tokens', 0)
    output_tokens = usage.get('output_tokens', 0)
    total_tokens = usage.get('total_tokens', 0)
    
    lines = []
    lines.append(f"Token Usage:")
    lines.append(f"  Input tokens:  {input_tokens:,}")
    lines.append(f"  Output tokens: {output_tokens:,}")
    lines.append(f"  Total tokens:  {total_tokens:,}")
    
    # Add cost estimation if model_id is provided
    if model_id and total_tokens > 0:
        cost = estimate_cost(input_tokens, output_tokens, model_id)
        if cost:
            lines.append(f"  Estimated cost: ${cost:.6f}")
    
    return "\n".join(lines)


def estimate_cost(input_tokens, output_tokens, model_id):
    """
    Estimate cost based on token usage and model.
    
    Uses actual token counts from the Bedrock API response.
    Pricing is per 1M tokens and varies by model:
    - Claude 3.5 Sonnet: $3.00/M input, $15.00/M output
    - Claude 3 Haiku: $0.25/M input, $1.25/M output
    - Claude 3 Sonnet: $3.00/M input, $15.00/M output
    - Claude 3 Opus: $15.00/M input, $75.00/M output
    
    Formula: (input_tokens / 1,000,000) * input_price + (output_tokens / 1,000,000) * output_price
    
    Args:
        input_tokens: Number of input tokens (from API response)
        output_tokens: Number of output tokens (from API response)
        model_id: Model identifier
    
    Returns:
        float: Estimated cost in USD, or None if pricing not available
    """
    if not model_id:
        return None
    
    model_id_lower = model_id.lower()
    
    # Determine provider and model type
    if 'anthropic' in model_id_lower or 'claude' in model_id_lower:
        provider = 'anthropic'
        if 'haiku' in model_id_lower:
            model_type = 'claude-3-haiku'
        elif 'sonnet' in model_id_lower:
            if '3-5' in model_id_lower or '3.5' in model_id_lower:
                model_type = 'claude-3-5-sonnet'
            else:
                model_type = 'claude-3-sonnet'
        elif 'opus' in model_id_lower:
            model_type = 'claude-3-opus'
        else:
            # Default to sonnet pricing
            model_type = 'claude-3-sonnet'
    elif 'mistral' in model_id_lower:
        provider = 'mistral'
        model_type = 'default'
    elif 'llama' in model_id_lower or 'meta' in model_id_lower:
        provider = 'meta'
        model_type = 'llama3'
    elif 'titan' in model_id_lower or 'amazon' in model_id_lower:
        provider = 'amazon'
        model_type = 'titan'
    elif 'cohere' in model_id_lower:
        provider = 'cohere'
        model_type = 'command-r'
    else:
        return None
    
    # Get pricing
    try:
        if provider in PRICING and model_type in PRICING[provider]:
            pricing = PRICING[provider][model_type]
            input_cost = (input_tokens / 1_000_000) * pricing['input']
            output_cost = (output_tokens / 1_000_000) * pricing['output']
            return input_cost + output_cost
    except (KeyError, TypeError):
        pass
    
    return None


def estimate_input_tokens(text):
    """
    Estimate input tokens from text (rough approximation).
    
    Uses a simple heuristic: ~4 characters per token for English text.
    This is a PRE-REQUEST estimate shown immediately to the user.
    
    NOTE: Actual token counts come from the Bedrock API response and may differ
    significantly from this estimate, especially for:
    - Non-English text
    - Code snippets
    - Special characters
    - Model-specific tokenization
    
    Args:
        text: Input text
    
    Returns:
        int: Estimated token count (approximate)
    """
    if not text:
        return 0
    # Rough estimate: ~4 characters per token for English
    # This is approximate and may vary significantly
    # Actual tokens are provided by the Bedrock API response
    return max(1, len(text) // 4)


def format_session_summary(session_totals, model_id=None):
    """
    Format session summary with total token usage.
    
    Args:
        session_totals: Dictionary with 'input_tokens', 'output_tokens', 'total_tokens', 'total_cost'
        model_id: Optional model ID for display
    
    Returns:
        str: Formatted summary string
    """
    input_tokens = session_totals.get('input_tokens', 0)
    output_tokens = session_totals.get('output_tokens', 0)
    total_tokens = session_totals.get('total_tokens', 0)
    total_cost = session_totals.get('total_cost', 0)
    
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("Session Summary")
    lines.append("=" * 70)
    if model_id:
        lines.append(f"Model: {model_id}")
    lines.append(f"Total Input tokens:  {input_tokens:,}")
    lines.append(f"Total Output tokens: {output_tokens:,}")
    lines.append(f"Total tokens:        {total_tokens:,}")
    if total_cost > 0:
        lines.append(f"Total Estimated cost: ${total_cost:.6f}")
    lines.append("=" * 70 + "\n")
    
    return "\n".join(lines)

