import subprocess
import re


class CopilotAPI:
    """Handles Copilot CLI interactions."""

    def __init__(self, api_config):
        """Initialize Copilot API."""
        self.api_config = api_config

    @staticmethod
    def get_response(query, debug=False):
        """Fetch response from Copilot CLI."""
        try:
            cmd = ["gh", "copilot", "explain", query]
            result = subprocess.run(cmd, text=True, capture_output=True, timeout=10)

            if result.returncode == 0:
                raw_output = result.stdout.strip()
                if debug:
                    print(f"\n[DEBUG] Raw Output:\n{raw_output}")

                # Extract explanation
                match = re.search(r"Explanation:\s*(.*)", raw_output, re.DOTALL)
                return match.group(1).strip() if match else "No explanation found."
            else:
                return f"Error: {result.stderr.strip()}"
        except FileNotFoundError:
            return "Error: GitHub CLI ('gh') not found. Install and configure it."
        except subprocess.TimeoutExpired:
            return "Error: Copilot command timed out."
        except Exception as e:
            return f"Unexpected error: {e}"
