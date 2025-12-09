import time
from google import genai

class GeminiAPI:
    """Class for Gemini API integration."""

    def __init__(self, config):
        """Initialize the API with given configuration."""
        self.gemini_client = None
        self.api_key = config.get("api_key")
        self.gemini_model = config.get("model")

    def get_gemini_client(self):
        """
        Get the Gemini client instance.
        """
        if not self.gemini_client:
            self.gemini_client = genai.Client(api_key=self.api_key)
        return self.gemini_client

    def get_response(self, query, retry_count=0, max_retries=3, debug=False):
        """Fetch AI response from Gemini API."""
        if not self.gemini_client:
            self.gemini_client = self.get_gemini_client()

        try:
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=query
            )
            return response.text

        except Exception as e:  # Catch ALL exceptions for now
            print(f"❌ Gemini API Error: {e}")  # Log the FULL exception for debugging
            if "RESOURCE_EXHAUSTED" in str(e):  # Check if the message contains the string
                print("Detected RESOURCE_EXHAUSTED in exception message.")
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count * 5
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    return self.get_response(query, retry_count + 1, max_retries)
                else:
                    print(f"❌ Max retries reached for Gemini API (RESOURCE_EXHAUSTED). Giving up.")
                    return ""
            else:  # If it is some other exception, we do not retry
                print(f"❌ Other Gemini API Error. Not retrying.")
                return ""
