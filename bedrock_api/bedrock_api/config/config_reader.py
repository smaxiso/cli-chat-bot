"""
Configuration reader for Bedrock API.
"""
import json
import os
from pathlib import Path


class ConfigReader:
    """Utility class to read API configurations from a JSON file."""
    
    def __init__(self, config_path=None):
        """
        Initialize ConfigReader.
        
        Args:
            config_path: Path to config file. If None, uses default location.
        """
        if config_path is None:
            # Default to config/config.json relative to project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.json"
        
        self.config_path = Path(config_path)
    
    def get_config(self):
        """
        Fetch all configurations from config file.
        
        Returns:
            dict: Configuration dictionary, empty dict if file not found or invalid
        """
        try:
            with open(self.config_path, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Error: Config file not found at {self.config_path}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format in config file: {e}")
            return {}
    
    def get_aws_config(self):
        """
        Fetch AWS configuration section.
        
        Returns:
            dict: AWS configuration, empty dict if not found
        """
        config = self.get_config()
        return config.get("aws", {})
    
    def get_bedrock_config(self):
        """
        Fetch Bedrock configuration section.
        
        Returns:
            dict: Bedrock configuration, empty dict if not found
        """
        config = self.get_config()
        return config.get("bedrock", {})
    
    def get_sso_config(self):
        """
        Fetch SSO configuration section.
        
        Returns:
            dict: SSO configuration, empty dict if not found
        """
        config = self.get_config()
        return config.get("sso", {})

