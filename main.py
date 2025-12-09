import argparse
import logging
import sys
import os
from datetime import datetime
from typing import List, Dict
from utils.tts_manager import TTSManager
from utils.util_methods import UtilMethods
from utils.config_reader import ConfigReader
from utils.terminal_formatter import TerminalFormatter

# Add bedrock_api to path to import ConversationManager
bedrock_api_path = os.path.join(os.path.dirname(__file__), 'bedrock_api')
if bedrock_api_path not in sys.path:
    sys.path.insert(0, bedrock_api_path)

try:
    from bedrock_api.services.conversation_manager import ConversationManager
except ImportError:
    ConversationManager = None
    logging.warning("ConversationManager not available - conversation context will not be shared across APIs")


class AIChatbot:
    """AI Chatbot with support for multiple APIs and TTS."""

    def __init__(self, api_name=None, enable_tts=False, debug=False, rate=150, volume=0.5, enable_stream=True, model=None):
        """Initialize the chatbot, prompting for API selection if needed."""
        self.api_name = None
        self.api_config = None
        self.api_instance = None
        self.rate = rate
        self.debug = debug
        self.volume = volume
        self.enable_tts = enable_tts  # TTS disabled by default
        self.enable_stream = enable_stream  # Streaming enabled by default
        self.model = model
        self.tts_manager = TTSManager(rate=self.rate, volume=self.volume) if self.enable_tts else None

        # Conversation session management
        self.conversations_dir = os.path.join(os.path.dirname(__file__), "conversations")
        os.makedirs(self.conversations_dir, exist_ok=True)
        self.current_conversation_name = None
        self.current_conversation_file = None

        # Create shared ConversationManager for all APIs
        # This ensures conversation context is preserved across API switches
        if ConversationManager:
            self.shared_conversation_manager = ConversationManager(
                max_history=None,  # No message limit
                max_tokens=160000,  # 80% of 200K context window
                trim_strategy="smart",
                reserve_tokens=1000
            )
        else:
            self.shared_conversation_manager = None

        # Store API instances to preserve conversation context when switching
        self.api_instances = {}  # Maps api_name -> api_instance

        # Load API classes dynamically
        self.config = ConfigReader.get_config()
        self.API_CLASSES = UtilMethods.get_api_classes(self.config)

        # Handle conversation session on startup
        self._handle_conversation_startup()

        # Select API
        self.select_api(api_name)

    def select_api(self, api_name=None):
        """Handles API selection and initialization."""
        if not api_name:
            api_name = UtilMethods.prompt_api_selection(API_CLASSES=self.API_CLASSES, chatbot_instance=self)
            if not api_name:
                self.exit_program("No API selected. Exiting the chatbot.")

        # Check if we already have an instance for this API (preserve context)
        if api_name in self.api_instances:
            self.api_name = api_name
            self.api_instance = self.api_instances[api_name]
            print(TerminalFormatter.format_system_message(f"\nSwitched to {self.api_name.capitalize()} API (conversation context preserved)."))
            return

        # New API - create instance
        self.api_name = api_name
        self.api_config = ConfigReader.get_api_config(api_name)

        if not self.api_config:
            self.exit_program(f"Error: Configuration for '{api_name}' is missing.")

        if self.model:
            if "model" in self.api_config:
                print(TerminalFormatter.format_system_message(f"Overriding default model with: {self.model}"))
                self.api_config["model"] = self.model
            elif "model_id" in self.api_config:
                print(TerminalFormatter.format_system_message(f"Overriding default model_id with: {self.model}"))
                self.api_config["model_id"] = self.model

        try:
            # Try to initialize with shared conversation manager
            api_class = self.API_CLASSES[api_name]
            
            # Check if API class accepts conversation_manager parameter
            import inspect
            sig = inspect.signature(api_class.__init__)
            
            if self.shared_conversation_manager and 'conversation_manager' in sig.parameters:
                # API accepts conversation_manager, pass it
                self.api_instance = api_class(self.api_config, conversation_manager=self.shared_conversation_manager)
            elif self.shared_conversation_manager and 'debug' in sig.parameters and 'conversation_manager' in sig.parameters:
                # API accepts both debug and conversation_manager
                self.api_instance = api_class(self.api_config, debug=self.debug, conversation_manager=self.shared_conversation_manager)
            else:
                # API doesn't accept conversation_manager, initialize normally
                if 'debug' in sig.parameters:
                    self.api_instance = api_class(self.api_config, debug=self.debug)
                else:
                    self.api_instance = api_class(self.api_config)
                
                # Try to set conversation_manager as attribute after initialization
                if self.shared_conversation_manager:
                    if hasattr(self.api_instance, 'conversation_manager'):
                        self.api_instance.conversation_manager = self.shared_conversation_manager
                    elif hasattr(self.api_instance, 'client') and hasattr(self.api_instance.client, 'conversation_manager'):
                        self.api_instance.client.conversation_manager = self.shared_conversation_manager
            
            # Store instance to preserve context when switching back
            self.api_instances[api_name] = self.api_instance
        except KeyError:
            self.exit_program(f"Error: API '{api_name}' is not supported.")
        except Exception as e:
            self.exit_program(f"Error initializing API '{api_name}': {e}")

        print(TerminalFormatter.format_system_message(f"\nUsing {self.api_name.capitalize()} API."))
    
    def _handle_conversation_startup(self):
        """Handle conversation selection on startup."""
        conversations = self._list_conversations()
        
        # Show startup menu
        print("\n" + "=" * 70)
        print("Welcome! What would you like to do?")
        print("=" * 70)
        if conversations:
            print("1. Continue last conversation")
            print("2. Start new conversation")
            print("3. Load specific conversation")
        else:
            print("1. Start new conversation")
        print("=" * 70)
        
        while True:
            try:
                if conversations:
                    choice = input("\nEnter your choice (1-3): ").strip()
                else:
                    choice = input("\nEnter your choice (1): ").strip()
                
                if choice == "1":
                    if conversations:
                        # Continue last conversation (most recently updated)
                        last_conv = max(conversations, key=lambda x: x["last_updated"])
                        self._load_conversation(last_conv["name"])
                        print(TerminalFormatter.format_system_message(f"\n✓ Loaded conversation: {last_conv['name']}\n"))
                    else:
                        # Start new conversation (no existing conversations)
                        name = input("Enter conversation name (or press Enter for auto-generated name): ").strip()
                        if not name:
                            name = f"chat_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
                        self._start_new_conversation(name)
                        print(TerminalFormatter.format_system_message(f"\n✓ Started new conversation: {name}\n"))
                    break
                
                elif choice == "2" and conversations:
                    # Start new conversation
                    name = input("Enter conversation name (or press Enter for auto-generated name): ").strip()
                    if not name:
                        name = f"chat_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
                    self._start_new_conversation(name)
                    print(TerminalFormatter.format_system_message(f"\n✓ Started new conversation: {name}\n"))
                    break
                
                elif choice == "3" and conversations:
                    # Load specific conversation
                    print("\nAvailable conversations:")
                    for i, conv in enumerate(conversations, 1):
                        msg_count = conv.get("message_count", 0)
                        print(f"  {i}. {conv['name']} ({msg_count} messages)")
                    print("  b. Back to main menu")
                    
                    while True:
                        conv_input = input("\nEnter conversation number (or 'b' to go back): ").strip().lower()
                        
                        if conv_input == 'b' or conv_input == 'back':
                            # Go back to main menu
                            print()  # Add spacing
                            break
                        
                        try:
                            conv_num = int(conv_input)
                            conv_index = conv_num - 1
                            if 0 <= conv_index < len(conversations):
                                self._load_conversation(conversations[conv_index]["name"])
                                print(TerminalFormatter.format_system_message(f"\n✓ Loaded conversation: {conversations[conv_index]['name']}\n"))
                                return  # Exit both loops
                            else:
                                print(TerminalFormatter.format_error("Invalid number. Please try again."))
                        except ValueError:
                            print(TerminalFormatter.format_error("Invalid input. Please enter a number or 'b' to go back."))
                    
                    # If we're here, user went back - show menu again
                    print("\n" + "=" * 70)
                    print("Welcome! What would you like to do?")
                    print("=" * 70)
                    print("1. Continue last conversation")
                    print("2. Start new conversation")
                    print("3. Load specific conversation")
                    print("=" * 70)
                    continue  # Continue outer loop to show menu again
                
                else:
                    if conversations:
                        print(TerminalFormatter.format_error("Invalid choice. Please enter 1, 2, or 3."))
                    else:
                        print(TerminalFormatter.format_error("Invalid choice. Please enter 1."))
            
            except (EOFError, KeyboardInterrupt):
                # Default to new conversation if interrupted
                name = f"chat_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
                self._start_new_conversation(name)
                break
    
    def _get_conversation_filepath(self, name: str) -> str:
        """Get filepath for a conversation by name."""
        # Sanitize name for filename
        safe_name = "".join(c for c in name if c.isalnum() or c in ('-', '_', ' ')).strip()
        safe_name = safe_name.replace(' ', '_')
        return os.path.join(self.conversations_dir, f"{safe_name}.json")
    
    def _list_conversations(self) -> List[Dict]:
        """List all saved conversations."""
        conversations = []
        
        if not os.path.exists(self.conversations_dir):
            return conversations
        
        for filename in os.listdir(self.conversations_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.conversations_dir, filename)
                try:
                    # Try to load to get metadata
                    manager = ConversationManager.load_from_file(filepath)
                    if manager:
                        conversations.append({
                            "name": filename[:-5],  # Remove .json extension
                            "filepath": filepath,
                            "message_count": len(manager.conversation_history),
                            "last_updated": manager.last_updated
                        })
                except Exception:
                    continue
        
        # Sort by last_updated (most recent first)
        conversations.sort(key=lambda x: x["last_updated"], reverse=True)
        return conversations
    
    def _start_new_conversation(self, name: str):
        """Start a new conversation with given name."""
        # Save current conversation if exists
        if self.current_conversation_name and self.shared_conversation_manager:
            self._save_current_conversation()
        
        # Create new conversation manager
        if ConversationManager:
            self.shared_conversation_manager = ConversationManager(
                max_history=None,
                max_tokens=160000,
                trim_strategy="smart",
                reserve_tokens=1000
            )
        
        self.current_conversation_name = name
        self.current_conversation_file = self._get_conversation_filepath(name)
        
        # Update all existing API instances to use the new conversation manager
        self._update_all_api_instances_conversation_manager()
        
        # Save empty conversation
        if self.shared_conversation_manager:
            self._save_current_conversation()
    
    def _load_conversation(self, name: str):
        """Load a conversation by name."""
        filepath = self._get_conversation_filepath(name)
        
        if not os.path.exists(filepath):
            print(TerminalFormatter.format_error(f"Conversation '{name}' not found."))
            self._start_new_conversation(name)
            return
        
        # Save current conversation if exists
        if self.current_conversation_name and self.shared_conversation_manager:
            self._save_current_conversation()
        
        # Load conversation
        if ConversationManager:
            manager = ConversationManager.load_from_file(filepath)
            if manager:
                self.shared_conversation_manager = manager
                self.current_conversation_name = name
                self.current_conversation_file = filepath
                
                # Update all existing API instances to use the loaded conversation manager
                self._update_all_api_instances_conversation_manager()
            else:
                print(TerminalFormatter.format_error(f"Failed to load conversation '{name}'. Starting new one."))
                self._start_new_conversation(name)
        else:
            self._start_new_conversation(name)
    
    def _update_all_api_instances_conversation_manager(self):
        """Update all existing API instances to use the current conversation manager."""
        if not self.shared_conversation_manager:
            return
        
        for api_name, api_instance in self.api_instances.items():
            # Update conversation_manager attribute if it exists (direct attribute)
            if hasattr(api_instance, 'conversation_manager'):
                api_instance.conversation_manager = self.shared_conversation_manager
            
            # Update nested conversation_manager (e.g., in client attribute)
            # This handles BedrockClaudeAdvancedAPI which has self.client.conversation_manager
            if hasattr(api_instance, 'client'):
                if hasattr(api_instance.client, 'conversation_manager'):
                    api_instance.client.conversation_manager = self.shared_conversation_manager
                # Also check for _conversation_manager (private attribute in BedrockClient)
                elif hasattr(api_instance.client, '_conversation_manager'):
                    api_instance.client._conversation_manager = self.shared_conversation_manager
            
            # For BedrockClaudeAdvancedAPI, also update the client's internal manager
            if hasattr(api_instance, '_bedrock_client') and hasattr(api_instance._bedrock_client, '_conversation_manager'):
                api_instance._bedrock_client._conversation_manager = self.shared_conversation_manager
    
    def _save_current_conversation(self):
        """Save current conversation to disk."""
        if self.shared_conversation_manager and self.current_conversation_file:
            try:
                self.shared_conversation_manager.save_to_file(self.current_conversation_file)
            except Exception as e:
                if self.debug:
                    logging.error(f"Failed to save conversation: {e}")

    def run(self):
        """Main interactive loop."""
        print(TerminalFormatter.format_system_message("\nWelcome to the AI Chatbot with TTS!"))
        empty_input_count = 0
        max_empty_retries = 3  # Set a limit for empty inputs
        first_query = True

        while True:
            try:
                if first_query:
                    print(TerminalFormatter.format_system_message("\nCommands: Type /help for available commands"))
                    print(TerminalFormatter.format_system_message(f"Streaming: {'ON' if self.enable_stream else 'OFF'}"))
                    print(TerminalFormatter.format_system_message(f"Text-to-Speech: {'ON' if self.enable_tts else 'OFF'}"))
                    formatted_prompt = TerminalFormatter.format_api_name(self.api_name) + " Enter your query: "
                    query = input(formatted_prompt).strip()
                else:
                    formatted_prompt = TerminalFormatter.format_api_name(self.api_name) + " Enter your query: "
                    query = input(formatted_prompt).strip()

                if not query:
                    empty_input_count += 1
                    print(TerminalFormatter.format_error(
                        f"No input detected. Please try again. ({empty_input_count}/{max_empty_retries})"))

                    if empty_input_count >= max_empty_retries:
                        self.exit_program("Too many empty inputs. Exiting the chatbot.")
                    continue

                empty_input_count = 0  # Reset counter on valid input

                # Commands must start with "/" prefix
                query_clean = query.strip().lower()

                # Check for exit/quit commands (with / prefix)
                if query_clean in ['/exit', '/quit']:
                    # Show conversation summaries for all APIs that have stats
                    summaries_shown = False
                    for api_name, api_instance in self.api_instances.items():
                        if hasattr(api_instance, 'get_conversation_stats'):
                            stats = api_instance.get_conversation_stats()
                            if stats.get('session_total_tokens', 0) > 0:
                                if not summaries_shown:
                                    print("\n" + "=" * 70)
                                    print("Session Summaries (All APIs)")
                                    print("=" * 70)
                                    summaries_shown = True
                                print(f"\n{api_name.capitalize()}:")
                                print(f"  Model: {stats.get('model_id', 'N/A')}")
                                print(f"  Total Input tokens:  {stats.get('session_input_tokens', 0):,}")
                                print(f"  Total Output tokens: {stats.get('session_output_tokens', 0):,}")
                                print(f"  Total tokens:        {stats.get('session_total_tokens', 0):,}")
                                if stats.get('session_total_cost', 0) > 0:
                                    print(f"  Total Estimated cost: ${stats.get('session_total_cost', 0):.6f}")
                    if summaries_shown:
                        print("=" * 70)
                    self.exit_program("Exiting the app. Goodbye!")

                if query_clean == '/help':
                    self.show_help()
                    continue

                if query_clean == '/switch':
                    # Show conversation summary before switching (if available)
                    if hasattr(self.api_instance, 'get_conversation_stats'):
                        stats = self.api_instance.get_conversation_stats()
                        if stats.get('session_total_tokens', 0) > 0:
                            print("\n" + "=" * 70)
                            print(f"Session Summary ({self.api_name})")
                            print("=" * 70)
                            print(f"Model: {stats.get('model_id', 'N/A')}")
                            print(f"Total Input tokens:  {stats.get('session_input_tokens', 0):,}")
                            print(f"Total Output tokens: {stats.get('session_output_tokens', 0):,}")
                            print(f"Total tokens:        {stats.get('session_total_tokens', 0):,}")
                            if stats.get('session_total_cost', 0) > 0:
                                print(f"Total Estimated cost: ${stats.get('session_total_cost', 0):.6f}")
                            print("=" * 70)
                    print(TerminalFormatter.format_system_message("\nSwitching API..."))
                    print(TerminalFormatter.format_system_message("Note: Conversation context is preserved when switching back."))
                    self.select_api()
                    first_query = True
                    continue

                if query_clean in ['/stream', '/stream-on', '/stream-off', '/no-stream']:
                    if query_clean in ['/stream', '/stream-on']:
                        self.enable_stream = True
                    else:
                        self.enable_stream = False
                    print(TerminalFormatter.format_system_message(f"\nStreaming: {'ON' if self.enable_stream else 'OFF'}\n"))
                    continue

                if query_clean in ['/tts', '/tts-on', '/tts-off', '/no-tts']:
                    if query_clean in ['/tts', '/tts-on']:
                        if self.enable_tts:
                            print(TerminalFormatter.format_system_message("\n✓ Text-to-Speech service is already active.\n"))
                        else:
                            self.enable_tts = True
                            if self.tts_manager is None:
                                from utils.tts_manager import TTSManager
                                self.tts_manager = TTSManager(rate=self.rate, volume=self.volume)
                            print(TerminalFormatter.format_system_message("\n✓ Text-to-Speech service has been started. Responses will be spoken from now onwards.\n"))
                    else:  # /tts-off or /no-tts
                        if not self.enable_tts:
                            print(TerminalFormatter.format_system_message("\n✓ Text-to-Speech service is already not active.\n"))
                        else:
                            self.enable_tts = False
                            if self.tts_manager:
                                self.tts_manager.stop_tts()
                            print(TerminalFormatter.format_system_message("\n✓ Text-to-Speech service has been stopped.\n"))
                    continue

                if query_clean == '/stats':
                    if hasattr(self.api_instance, 'get_conversation_stats'):
                        stats = self.api_instance.get_conversation_stats()
                        print("\n" + "=" * 70)
                        print("Conversation Statistics")
                        print("=" * 70)
                        print(f"Total messages: {stats.get('total_messages', 0)}")
                        print(f"  - User messages: {stats.get('user_messages', 0)}")
                        print(f"  - Assistant messages: {stats.get('assistant_messages', 0)}")
                        print(f"  - System messages: {stats.get('system_messages', 0)}")
                        if stats.get('estimated_tokens'):
                            print(f"\nEstimated tokens in history: {stats.get('estimated_tokens', 0):,}")
                            if stats.get('max_tokens'):
                                usage_pct = (stats.get('estimated_tokens', 0) / stats.get('max_tokens', 1)) * 100
                                print(f"Max tokens limit: {stats.get('max_tokens', 0):,}")
                                print(f"Token usage: {usage_pct:.1f}%")
                        print(f"\nSession Input tokens:  {stats.get('session_input_tokens', 0):,}")
                        print(f"Session Output tokens: {stats.get('session_output_tokens', 0):,}")
                        print(f"Session Total tokens:  {stats.get('session_total_tokens', 0):,}")
                        if stats.get('session_total_cost', 0) > 0:
                            print(f"Session Total cost: ${stats.get('session_total_cost', 0):.6f}")
                        print("=" * 70 + "\n")
                    else:
                        print(TerminalFormatter.format_system_message("\nConversation statistics not available for this API.\n"))
                    continue

                if query_clean == '/clear':
                    # Clear shared conversation manager if available
                    if self.shared_conversation_manager:
                        count = self.shared_conversation_manager.clear()
                        self._save_current_conversation()  # Save after clearing
                        print(TerminalFormatter.format_system_message(f"\n✓ Conversation history cleared ({count} messages removed)\n"))
                    elif hasattr(self.api_instance, 'clear_conversation'):
                        count = self.api_instance.clear_conversation()
                        print(TerminalFormatter.format_system_message(f"\n✓ Conversation history cleared ({count} messages removed)\n"))
                    elif hasattr(self.api_instance, 'conversation_manager') and hasattr(self.api_instance.conversation_manager, 'clear'):
                        count = self.api_instance.conversation_manager.clear()
                        print(TerminalFormatter.format_system_message(f"\n✓ Conversation history cleared ({count} messages removed)\n"))
                    else:
                        print(TerminalFormatter.format_system_message("\nConversation history not available for this API.\n"))
                    continue

                if query_clean.startswith('/new-chat'):
                    # /new-chat [name] - Start new conversation
                    parts = query.split(maxsplit=1)
                    if len(parts) > 1:
                        name = parts[1].strip()
                    else:
                        name = f"chat_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
                    
                    self._start_new_conversation(name)
                    print(TerminalFormatter.format_system_message(f"\n✓ Started new conversation: {name}\n"))
                    continue

                if query_clean.startswith('/switch-chat'):
                    # /switch-chat <name> - Switch to different conversation
                    parts = query.split(maxsplit=1)
                    if len(parts) > 1:
                        name = parts[1].strip()
                        self._load_conversation(name)
                        print(TerminalFormatter.format_system_message(f"\n✓ Switched to conversation: {name}\n"))
                    else:
                        print(TerminalFormatter.format_error("\nError: Please provide a conversation name. Example: /switch-chat work-project\n"))
                    continue

                if query_clean == '/list-chats':
                    # List all conversations
                    conversations = self._list_conversations()
                    if conversations:
                        print("\n" + "=" * 70)
                        print("Available Conversations")
                        print("=" * 70)
                        for i, conv in enumerate(conversations, 1):
                            current_marker = " (current)" if conv["name"] == self.current_conversation_name else ""
                            print(f"  {i}. {conv['name']}{current_marker}")
                            print(f"     Messages: {conv['message_count']}, Last updated: {conv['last_updated'].strftime('%Y-%m-%d %H:%M:%S')}")
                        print("=" * 70 + "\n")
                    else:
                        print(TerminalFormatter.format_system_message("\nNo saved conversations found.\n"))
                    continue

                if query_clean.startswith('/rename-chat'):
                    # /rename-chat <new-name> - Rename current conversation
                    if not self.current_conversation_name:
                        print(TerminalFormatter.format_error("\nError: No active conversation to rename.\n"))
                        continue
                    
                    parts = query.split(maxsplit=1)
                    if len(parts) > 1:
                        new_name = parts[1].strip()
                        if new_name:
                            # Delete old file
                            old_file = self.current_conversation_file
                            if os.path.exists(old_file):
                                os.remove(old_file)
                            
                            # Update name and filepath
                            self.current_conversation_name = new_name
                            self.current_conversation_file = self._get_conversation_filepath(new_name)
                            
                            # Save with new name
                            self._save_current_conversation()
                            print(TerminalFormatter.format_system_message(f"\n✓ Conversation renamed to: {new_name}\n"))
                        else:
                            print(TerminalFormatter.format_error("\nError: Please provide a new name. Example: /rename-chat new-name\n"))
                    else:
                        print(TerminalFormatter.format_error("\nError: Please provide a new name. Example: /rename-chat new-name\n"))
                    continue

                # Echo the query with formatting
                print(TerminalFormatter.format_query(f"\nYour query: {query}"))

                if self.tts_manager:
                    self.tts_manager.stop_tts()

                # Get response with streaming support
                response = self._get_response_with_streaming(query)
                
                # Only print formatted response if not streaming (streaming already printed)
                if not (self.enable_stream and hasattr(self.api_instance, 'get_response_stream')):
                    print(TerminalFormatter.format_response(response, self.api_name))

                # Auto-save conversation after each exchange
                if self.shared_conversation_manager:
                    self._save_current_conversation()

                if self.enable_tts:
                    self.tts_manager.speak_text(response)
                first_query = False

            except EOFError:
                self.exit_program("Exiting the app due to EOF.")
            except KeyboardInterrupt:
                print(TerminalFormatter.format_system_message("\nInput interrupted. Use /exit to quit.\n"))
                continue

    def _get_response_with_streaming(self, query):
        """
        Get response with streaming support if available.
        
        Args:
            query: User query
        
        Returns:
            str: Complete response text
        """
        # Check if API supports streaming
        if self.enable_stream and hasattr(self.api_instance, 'get_response_stream'):
            try:
                # Use streaming if available
                response_text = ""
                print("\n" + "=" * 70)
                print(f"{self.api_name.capitalize()} Response:")
                print("=" * 70)
                
                for event in self.api_instance.get_response_stream(query, debug=self.debug):
                    if 'chunk' in event:
                        chunk = event['chunk']
                        print(chunk, end='', flush=True)
                        response_text += chunk
                    elif 'usage' in event and self.debug:
                        usage = event['usage']
                        logging.debug(f"Token usage: {usage}")
                
                print("\n" + "=" * 70)
                return response_text
            except Exception as e:
                logging.warning(f"Streaming failed, falling back to non-streaming: {e}")
                # Fall back to non-streaming
                return self.api_instance.get_response(query, debug=self.debug)
        else:
            # Non-streaming mode
            return self.api_instance.get_response(query, debug=self.debug)

    @staticmethod
    def show_help():
        """Display help instructions."""
        help_text = """
    Commands (must start with "/" prefix):
    - /exit, /quit         - Exit the application
    - /help                - Show this help message
    - /switch              - Change the API at runtime
    - /stream, /stream-on  - Enable streaming responses
    - /stream-off, /no-stream - Disable streaming responses
    - /tts, /tts-on        - Enable text-to-speech
    - /tts-off, /no-tts    - Disable text-to-speech
    - /new-chat [name]     - Start new conversation (optional name)
    - /switch-chat <name>  - Switch to different conversation
    - /list-chats          - List all saved conversations
    - /rename-chat <name>  - Rename current conversation
    - /stats               - Show conversation statistics (bedrock_claude_advanced only)
    - /clear               - Clear conversation history (bedrock_claude_advanced only)

    Command-line arguments:
    - --api <name>         - Select AI API
    - --tts                - Enable text-to-speech (TTS is OFF by default)
    - --no-stream          - Disable streaming (streaming is ON by default)
    - --debug              - Enable verbose logging
    - --model <model_id>   - Specify model (for APIs that support multiple models)

    Available APIs:
    - copilot: GitHub Copilot CLI
    - claude: Anthropic Claude API
    - gemini: Google Gemini API
    - openai: OpenAI GPT models API
    - bedrock_claude: Claude via AWS Bedrock (legacy)
    - bedrock_claude_advanced: Claude via AWS Bedrock with conversation management
    - ollama: Local Ollama models (streaming, context management)

    Note: All commands must start with "/" to be recognized.
          All other text is sent to the AI as prompts.
    """
        print(TerminalFormatter.format_help_text(help_text))

    @staticmethod
    def exit_program(message):
        """Centralized exit function."""
        print(TerminalFormatter.format_system_message(message))
        exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Chatbot with TTS.")
    parser.add_argument("--api", type=str, choices=["copilot", "claude", "gemini", "openai", "bedrock_claude", "bedrock_claude_advanced", "ollama"], help="Select AI API.")
    parser.add_argument("--tts", action="store_true", dest="enable_tts", help="Enable text-to-speech (TTS is OFF by default).")
    parser.add_argument("--no-stream", action="store_false", dest="enable_stream", help="Disable streaming (streaming is ON by default).")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for verbose logging.")
    parser.add_argument("--rate", type=int, default=150, help="Set TTS rate (default 150).")
    parser.add_argument("--volume", type=float, default=0.5, help="Set TTS volume (default 0.5).")
    parser.add_argument("--model", type=str, help="Specify the model to use (overrides config)")

    args = parser.parse_args()

    chatbot = AIChatbot(
        api_name=args.api,
        enable_tts=args.enable_tts,
        enable_stream=args.enable_stream,
        debug=args.debug,
        rate=args.rate,
        volume=args.volume,
        model=args.model
    )
    chatbot.run()
