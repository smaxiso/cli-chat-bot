import openai


class OpenAIAPI:
    """Handles requests to the OpenAI API."""

    def __init__(self, config):
        """Initialize the API with given configuration."""
        self.api_key = config.get("api_key")
        self.model = config.get("model", "gpt-4o")
        self.client = openai.OpenAI(api_key=self.api_key)

    def get_response(self, prompt, debug=False):
        """Fetch AI response from OpenAI API."""
        try:
            if debug:
                print(f"Sending request to OpenAI API using model: {self.model}")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Provide clear and concise answers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )

            if debug:
                print(f"OpenAI response received: {len(response.choices[0].message.content)} characters")

            return response.choices[0].message.content
        except Exception as e:
            error_message = f"OpenAI API Error: {str(e)}"
            if debug:
                print(error_message)
            return error_message
