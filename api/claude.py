import anthropic

class ClaudeAPI:
    """Handles requests to the Claude AI API."""

    def __init__(self, config):
        """Initialize the API with given configuration."""
        self.api_key = config.get("api_key")
        self.endpoint = config.get("endpoint")
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def get_response(self, prompt, debug=False):
        """Fetch AI response from Claude API."""
        try:
            message = self.client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=500,
                temperature=0.7,
                system="You are an AI assistant. Provide clear and concise answers.",
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            )
            return message.content
        except Exception as e:
            return f"Error: {e}"
