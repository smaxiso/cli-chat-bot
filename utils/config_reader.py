import json
import os

class ConfigReader:
    """Utility class to read API configurations from a JSON file."""

    CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../resources/config.json")

    @staticmethod
    def get_config():
        """Fetch all API configurations."""
        try:
            with open(ConfigReader.CONFIG_PATH, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Error: Config file not found at {ConfigReader.CONFIG_PATH}")
            return {}
        except json.JSONDecodeError:
            print("Error: Invalid JSON format in config file.")
            return {}

    @staticmethod
    def get_api_config(api_name):
        """Fetch configuration for a specific API."""
        try:
            with open(ConfigReader.CONFIG_PATH, "r") as file:
                config = json.load(file)
                return config.get(api_name, {})
        except FileNotFoundError:
            print(f"Error: Config file not found at {ConfigReader.CONFIG_PATH}")
            return {}
        except json.JSONDecodeError:
            print("Error: Invalid JSON format in config file.")
            return {}
