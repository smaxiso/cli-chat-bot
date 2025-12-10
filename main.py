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
from colorama import Style

# Add bedrock_api to path to import ConversationManager
bedrock_api_path = os.path.join(os.path.dirname(__file__), 'bedrock_api')
if bedrock_api_path not in sys.path:
    sys.path.insert(0, bedrock_api_path)

try:
    from bedrock_api.services.conversation_manager import ConversationManager
except ImportError:
    ConversationManager = None
    logging.warning("ConversationManager not available - conversation context will not be shared across APIs")

try:
    from services.response_cache import ResponseCache
except ImportError:
    ResponseCache = None
    logging.warning("ResponseCache not available - response caching will be disabled")


class AIChatbot:
    """AI Chatbot with support for multiple APIs and TTS."""

    def __init__(self, api_name=None, enable_tts=False, debug=False, rate=150, volume=0.5, enable_stream=True, model=None, enable_cache=False):
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

        # Cache directory (separate from conversations)
        self.cache_dir = os.path.join(os.path.dirname(__file__), "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

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

        # Create shared ResponseCache for all APIs
        # This enables response caching (per-conversation, not context-aware)
        # Can be enabled via --cache argument or /cache-on command
        self.cache_enabled = enable_cache  # Default: disabled
        
        if ResponseCache and self.cache_enabled:
            cache_file = os.path.join(self.cache_dir, "response_cache.json")
            self.response_cache = ResponseCache(
                ttl_hours=24,
                max_size=1000,
                persistent=True,
                cache_file=cache_file
            )
            if self.debug:
                logging.info(f"ResponseCache initialized: {cache_file}")
        else:
            self.response_cache = None
            if self.debug:
                if not ResponseCache:
                    logging.warning("ResponseCache not available - caching disabled")
                elif not self.cache_enabled:
                    logging.info("ResponseCache disabled by user")

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
            
            # Check if API class accepts parameters
            import inspect
            sig = inspect.signature(api_class.__init__)
            
            # Build initialization arguments
            init_kwargs = {}
            
            # Add debug if API supports it
            if 'debug' in sig.parameters:
                init_kwargs['debug'] = self.debug
            
            # Add conversation_manager if API supports it
            if self.shared_conversation_manager and 'conversation_manager' in sig.parameters:
                init_kwargs['conversation_manager'] = self.shared_conversation_manager
            
            # Add cache if API supports it
            if self.response_cache and 'cache' in sig.parameters:
                init_kwargs['cache'] = self.response_cache
            
            # Initialize API instance
            if init_kwargs:
                self.api_instance = api_class(self.api_config, **init_kwargs)
            else:
                self.api_instance = api_class(self.api_config)
            
            # Try to set conversation_manager as attribute after initialization if not passed
            if self.shared_conversation_manager and 'conversation_manager' not in init_kwargs:
                if hasattr(self.api_instance, 'conversation_manager'):
                    self.api_instance.conversation_manager = self.shared_conversation_manager
                elif hasattr(self.api_instance, 'client') and hasattr(self.api_instance.client, 'conversation_manager'):
                    self.api_instance.client.conversation_manager = self.shared_conversation_manager
            
            # Try to set cache as attribute after initialization if not passed
            if self.response_cache and 'cache' not in init_kwargs:
                if hasattr(self.api_instance, 'cache'):
                    self.api_instance.cache = self.response_cache
                elif hasattr(self.api_instance, 'client') and hasattr(self.api_instance.client, 'cache'):
                    self.api_instance.client.cache = self.response_cache
            
            # Set conversation_id for per-conversation caching
            # Always set it, even if current_conversation_name is None (will use "default" in cache)
            conversation_id = self.current_conversation_name or "default"
            if hasattr(self.api_instance, '_conversation_id'):
                self.api_instance._conversation_id = conversation_id
            elif hasattr(self.api_instance, 'client') and hasattr(self.api_instance.client, '_conversation_id'):
                self.api_instance.client._conversation_id = conversation_id
            
            if self.debug:
                logging.debug(f"Set conversation_id on {api_name} API instance: {conversation_id}")
            
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
        
        # Prevent conflicts with cache file
        cache_filename = "response_cache.json"
        if f"{safe_name}.json" == cache_filename:
            safe_name = f"{safe_name}_conv"  # Append _conv to avoid conflict
        
        return os.path.join(self.conversations_dir, f"{safe_name}.json")
    
    def _list_conversations(self) -> List[Dict]:
        """List all saved conversations."""
        conversations = []
        
        if not os.path.exists(self.conversations_dir):
            return conversations
        
        # Exclude cache directory and any subdirectories
        cache_dir_name = os.path.basename(self.cache_dir)
        
        for filename in os.listdir(self.conversations_dir):
            # Skip directories (like cache directory if it somehow ends up here)
            filepath = os.path.join(self.conversations_dir, filename)
            if os.path.isdir(filepath):
                continue
            
            if filename.endswith('.json'):
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
        
        # Update conversation_id for all API instances (for per-conversation caching)
        self._update_all_api_instances_conversation_id()
        
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
                
                # Update conversation_id for all API instances (for per-conversation caching)
                self._update_all_api_instances_conversation_id()
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
    
    def _update_all_api_instances_conversation_id(self):
        """Update all existing API instances with current conversation ID for per-conversation caching."""
        conversation_id = self.current_conversation_name or "default"
        
        for api_name, api_instance in self.api_instances.items():
            # Set conversation_id for per-conversation caching
            if hasattr(api_instance, '_conversation_id'):
                api_instance._conversation_id = conversation_id
            elif hasattr(api_instance, 'client') and hasattr(api_instance.client, '_conversation_id'):
                api_instance.client._conversation_id = conversation_id
        
        if self.debug:
            logging.debug(f"Updated conversation_id on all API instances: {conversation_id}")
    
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
                
                # Add separator before user input
                print(TerminalFormatter.format_separator())
                
                # Capture input with [User]: prefix (user types directly after prefix)
                query = input(f"{TerminalFormatter.USER_COLOR}[User]: {Style.RESET_ALL}").strip()

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
                    # Show conversation statistics
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
                        print("=" * 70)
                    else:
                        print(TerminalFormatter.format_system_message("\nConversation statistics not available for this API.\n"))
                    
                    # Show cache statistics
                    if self.response_cache:
                        cache_stats = self.response_cache.get_stats()
                        print("\n" + "=" * 70)
                        print("Response Cache Statistics")
                        print("=" * 70)
                        print(f"Total entries: {cache_stats['total_entries']}")
                        print(f"Valid entries: {cache_stats['valid_entries']}")
                        print(f"Expired entries: {cache_stats['expired_entries']}")
                        print(f"Max size: {cache_stats['max_size']}")
                        print(f"TTL: {cache_stats['ttl_hours']} hours")
                        print(f"\nCache hits: {cache_stats['hits']}")
                        print(f"Cache misses: {cache_stats['misses']}")
                        print(f"Hit rate: {cache_stats['hit_rate_percent']}%")
                        print(f"Total sets: {cache_stats['sets']}")
                        print(f"Evictions: {cache_stats['evictions']}")
                        print("=" * 70 + "\n")
                    continue
                
                if query_clean == '/cache-clear':
                    if self.response_cache:
                        self.response_cache.clear()
                        print(TerminalFormatter.format_system_message("\n✓ Response cache cleared.\n"))
                    else:
                        print(TerminalFormatter.format_error("\nResponse cache is not available.\n"))
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

                if self.tts_manager:
                    self.tts_manager.stop_tts()

                # Display timestamp (input already shown with [User]: prefix)
                user_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"{TerminalFormatter.TIMESTAMP_COLOR}{user_timestamp}{Style.RESET_ALL}")
                print()  # New line after user message
                
                # Get response with streaming support
                api_name_display = self.api_name.capitalize()
                
                try:
                    # Check if streaming will be used
                    is_streaming = self.enable_stream and hasattr(self.api_instance, 'get_response_stream')
                    
                    response, is_cached = self._get_response_with_streaming(query)
                    
                    # Format and display model response with timestamp
                    # Only format if NOT streaming (streaming already printed with prefix and timestamp)
                    if not is_streaming or is_cached:
                        model_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        model_name = api_name_display
                        if is_cached:
                            model_name += " (cached)"
                        print(TerminalFormatter.format_model_message(response, model_name, model_timestamp))
                    
                    # Add separator at the end (for both streaming and non-streaming)
                    print(TerminalFormatter.format_separator())
                    print()  # New line after separator
                    print()  # Extra line for spacing before next Q-A pair
                except Exception as e:
                    raise

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
            tuple: (response_text, is_cached) - Complete response text and cache status
        """
        # Check if API supports streaming
        if self.enable_stream and hasattr(self.api_instance, 'get_response_stream'):
            try:
                # For streaming, check cache first to avoid unnecessary API calls
                # If cached, return immediately (don't stream cached responses)
                if hasattr(self.api_instance, 'cache') and self.api_instance.cache:
                    conversation_id = getattr(self.api_instance, '_conversation_id', "default")
                    cached = self.api_instance.cache.get(
                        query=query,
                        conversation_id=conversation_id,
                        model_id=self.api_instance.model,
                        temperature=0.7
                    )
                    if cached:
                        # Add to conversation history for cached response
                        if hasattr(self.api_instance, 'conversation_manager') and self.api_instance.conversation_manager:
                            self.api_instance.conversation_manager.add_user_message(query)
                            self.api_instance.conversation_manager.add_assistant_message(cached)
                        return cached, True
                
                # Cache miss - proceed with streaming
                # Start streaming with model name prefix
                api_name_display = self.api_name.capitalize()
                print(f"{TerminalFormatter.MODEL_COLOR}[{api_name_display}]: {Style.RESET_ALL}", end="", flush=True)
                
                response_text = ""
                for event in self.api_instance.get_response_stream(query, debug=self.debug):
                    if 'chunk' in event:
                        chunk = event['chunk']
                        print(chunk, end='', flush=True)
                        response_text += chunk
                    elif 'usage' in event and self.debug:
                        usage = event['usage']
                        logging.debug(f"Token usage: {usage}")
                
                # Add timestamp after streaming completes
                model_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n{TerminalFormatter.TIMESTAMP_COLOR}{model_timestamp}{Style.RESET_ALL}")
                return response_text, False
            except Exception as e:
                logging.warning(f"Streaming failed, falling back to non-streaming: {e}")
                # Fall back to non-streaming
                response, is_cached = self.api_instance.get_response(query, debug=self.debug)
                return response, is_cached
        else:
            # Non-streaming mode
            response, is_cached = self.api_instance.get_response(query, debug=self.debug)
            return response, is_cached

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
    - /stats               - Show conversation and cache statistics
    - /cache-clear         - Clear response cache
    - /cache-off, /no-cache - Disable response caching
    - /cache-on, /cache     - Enable response caching
    - /clear               - Clear conversation history

    Command-line arguments:
    - --api <name>         - Select AI API
    - --tts                - Enable text-to-speech (TTS is OFF by default)
    - --no-stream          - Disable streaming (streaming is ON by default)
    - --cache               - Enable response caching (caching is OFF by default)
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
    parser.add_argument("--cache", action="store_true", dest="enable_cache", help="Enable response caching (caching is disabled by default).")

    args = parser.parse_args()

    chatbot = AIChatbot(
        api_name=args.api,
        enable_tts=args.enable_tts,
        enable_stream=args.enable_stream,
        debug=args.debug,
        rate=args.rate,
        volume=args.volume,
        model=args.model,
        enable_cache=args.enable_cache
    )
    chatbot.run()
