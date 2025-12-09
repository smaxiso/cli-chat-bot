import importlib
from utils.terminal_formatter import TerminalFormatter  # Corrected import path

class UtilMethods:
    """Utility methods for the application"""

    @staticmethod
    def print_api_list(api_list):
        print(TerminalFormatter.format_system_message("\nAvailable AI APIs:"))
        for idx, api_name in enumerate(api_list, start=1):
            print(TerminalFormatter.format_system_message(f"{idx}. {api_name.capitalize()}"))

    @staticmethod
    def prompt_api_selection(API_CLASSES, chatbot_instance=None):
        """Prompt the user to select an API using number or name.
        
        Args:
            API_CLASSES: Dictionary of available API classes
            chatbot_instance: Optional AIChatbot instance for handling chat commands
        """
        api_list = list(API_CLASSES.keys())

        UtilMethods.print_api_list(api_list)

        while True:
            prompt = TerminalFormatter.format_prompt("\nSelect an API (number or name): ")
            selected = input(prompt).strip()

            # Check for chat management commands if chatbot_instance is provided
            if chatbot_instance:
                selected_lower = selected.lower()
                
                # Handle chat management commands
                if selected_lower.startswith('/new-chat'):
                    parts = selected.split(maxsplit=1)
                    if len(parts) > 1:
                        name = parts[1].strip()
                    else:
                        from datetime import datetime
                        name = f"chat_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
                    chatbot_instance._start_new_conversation(name)
                    print(TerminalFormatter.format_system_message(f"\n✓ Started new conversation: {name}\n"))
                    UtilMethods.print_api_list(api_list)
                    continue
                
                elif selected_lower.startswith('/switch-chat'):
                    parts = selected.split(maxsplit=1)
                    if len(parts) > 1:
                        name = parts[1].strip()
                        chatbot_instance._load_conversation(name)
                        print(TerminalFormatter.format_system_message(f"\n✓ Switched to conversation: {name}\n"))
                    else:
                        print(TerminalFormatter.format_error("\nError: Please provide a conversation name. Example: /switch-chat work-project\n"))
                    UtilMethods.print_api_list(api_list)
                    continue
                
                elif selected_lower == '/list-chats':
                    conversations = chatbot_instance._list_conversations()
                    if conversations:
                        print("\n" + "=" * 70)
                        print("Available Conversations")
                        print("=" * 70)
                        for i, conv in enumerate(conversations, 1):
                            current_marker = " (current)" if conv["name"] == chatbot_instance.current_conversation_name else ""
                            print(f"  {i}. {conv['name']}{current_marker}")
                            print(f"     Messages: {conv['message_count']}, Last updated: {conv['last_updated'].strftime('%Y-%m-%d %H:%M:%S')}")
                        print("=" * 70 + "\n")
                    else:
                        print(TerminalFormatter.format_system_message("\nNo saved conversations found.\n"))
                    UtilMethods.print_api_list(api_list)
                    continue
                
                elif selected_lower.startswith('/rename-chat'):
                    if not chatbot_instance.current_conversation_name:
                        print(TerminalFormatter.format_error("\nError: No active conversation to rename.\n"))
                    else:
                        parts = selected.split(maxsplit=1)
                        if len(parts) > 1:
                            new_name = parts[1].strip()
                            if new_name:
                                import os
                                old_file = chatbot_instance.current_conversation_file
                                if os.path.exists(old_file):
                                    os.remove(old_file)
                                chatbot_instance.current_conversation_name = new_name
                                chatbot_instance.current_conversation_file = chatbot_instance._get_conversation_filepath(new_name)
                                chatbot_instance._save_current_conversation()
                                print(TerminalFormatter.format_system_message(f"\n✓ Conversation renamed to: {new_name}\n"))
                            else:
                                print(TerminalFormatter.format_error("\nError: Please provide a new name. Example: /rename-chat new-name\n"))
                        else:
                            print(TerminalFormatter.format_error("\nError: Please provide a new name. Example: /rename-chat new-name\n"))
                    UtilMethods.print_api_list(api_list)
                    continue
            
            selected_lower = selected.lower()
            
            # If input is a number, get the corresponding API
            if selected_lower.isdigit():
                index = int(selected_lower) - 1
                if 0 <= index < len(api_list):
                    return api_list[index]
            # If input is a valid API name, return it
            elif selected_lower in API_CLASSES:
                return selected_lower
            elif selected_lower in ['exit', 'quit', '/exit', '/quit']:
                return None

            print(TerminalFormatter.format_error("Invalid selection. Please enter a valid number or API name."))
            print(TerminalFormatter.format_system_message("Type 'exit' to quit."))
            UtilMethods.print_api_list(api_list)

    @staticmethod
    def get_api_classes(config):
        """Dynamically load API classes based on config.json."""
        api_classes = {}
        import logging
        logger = logging.getLogger(__name__)

        for api_name, api_details in config.items():
            class_name = api_details.get("class_name")
            if class_name:
                try:
                    module = importlib.import_module(f"api.{api_name}")
                    api_classes[api_name] = getattr(module, class_name)
                except (ModuleNotFoundError, AttributeError) as e:
                    # Log as debug instead of printing error to reduce noise
                    # These are expected when optional dependencies aren't installed
                    logger.debug(f"API '{api_name}' not available - {e}")

        return api_classes
