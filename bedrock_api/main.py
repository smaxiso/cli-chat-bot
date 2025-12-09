#!/usr/bin/env python3
"""
Main entry point for Bedrock API application.
"""
import argparse
import sys
from bedrock_api.config.config_reader import ConfigReader
from bedrock_api.services.bedrock_client import BedrockClient
from bedrock_api.utils.token_usage import format_token_usage, estimate_input_tokens, format_session_summary, estimate_cost


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="AWS Bedrock API Client for Claude models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python main.py
  
  # Single prompt
  python main.py --prompt "What is AWS Bedrock?"
  
  # With custom model
  python main.py --prompt "Hello" --model "anthropic.claude-3-haiku-20240307-v1:0"
  
  # Debug mode
  python main.py --prompt "Test" --debug
  
  # List available models
  python main.py --list-models
  
  # List models by provider
  python main.py --list-models --provider "Anthropic"
        """
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        help="Single prompt to process (if not provided, runs in interactive mode)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model ID to use (overrides config default)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature parameter (0.0-1.0)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        dest="top_p",
        help="Top-p parameter (0.0-1.0)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        dest="max_tokens",
        help="Maximum tokens in response"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for verbose logging"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file (default: config/config.json)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available Gen AI models and exit"
    )
    parser.add_argument(
        "--provider",
        type=str,
        help="Filter models by provider (e.g., 'Anthropic', 'Amazon', 'AI21 Labs')"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming responses (shows response in real-time)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config_reader = ConfigReader(config_path=args.config)
        config = config_reader.get_config()
        
        if not config:
            print("Error: Failed to load configuration.")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Initialize Bedrock client
    try:
        client = BedrockClient(config, debug=args.debug)
    except Exception as e:
        print(f"Error initializing Bedrock client: {e}")
        sys.exit(1)
    
    # Store config reference for session refresh
    client._config = config
    
    # List models mode
    if args.list_models:
        try:
            print("\nFetching available Gen AI models (on-demand only)...")
            models = client.list_gen_ai_models(provider=args.provider, on_demand_only=True)
            print(client.format_models_list(models, show_on_demand=True))
            sys.exit(0)
        except Exception as e:
            print(f"Error listing models: {e}")
            sys.exit(1)
    
    # Single prompt mode
    if args.prompt:
        try:
            result = client.get_response(
                prompt=args.prompt,
                model_id=args.model,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                debug=args.debug
            )
            response_text = result['response']
            usage = result['usage']
            
            print("\n" + "=" * 70)
            print("Response:")
            print("=" * 70)
            print(response_text)
            print("=" * 70)
            print(f"\n{format_token_usage(usage, args.model or client.model_id)}")
            print("=" * 70 + "\n")
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            sys.exit(0)
        except Exception as e:
            print(f"\nError: {e}")
            sys.exit(1)
    
    # Interactive mode
    else:
        # Track current model selection (can be changed at runtime)
        current_model = args.model or client.model_id
        
        # Track session totals
        session_totals = {
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0,
            'total_cost': 0.0
        }
        
        # Track streaming mode (can be toggled at runtime)
        streaming_mode = args.stream
        
        # Track conversation context (enabled by default)
        use_context = True
        
        print("\n" + "=" * 70)
        print("AWS Bedrock API - Interactive Mode")
        print("=" * 70)
        print(f"Current model: {current_model}")
        print(f"Streaming mode: {'ON' if streaming_mode else 'OFF'}")
        print(f"Conversation context: {'ON' if use_context else 'OFF'} (maintains chat history)")
        print("\nCommands: Type /help for available commands")
        print("All other text is sent to the AI as prompts.\n")
        
        try:
            while True:
                try:
                    prompt = input("Prompt: ").strip()
                    
                    if not prompt:
                        continue
                    
                    # Commands must start with "/" prefix and be exact matches (case-insensitive)
                    # Strip whitespace and check exact match
                    prompt_clean = prompt.strip().lower()
                    
                    # Check for exit/quit commands (with / prefix)
                    if prompt_clean in ['/exit', '/quit']:
                        # Show session summary before exiting
                        if session_totals['total_tokens'] > 0:
                            print(format_session_summary(session_totals, current_model))
                        print("\nGoodbye!")
                        break
                    
                    if prompt_clean == '/help':
                        print("""
Commands (must start with "/" prefix):
  /exit, /quit         - Exit the application
  /help                - Show this help message
  /models              - List available Gen AI models
  /use-model <model_id> - Switch to a different model
  /current-model       - Show currently selected model
  /stream              - Toggle streaming mode (ON/OFF)
  /stream-on           - Enable streaming mode
  /stream-off          - Disable streaming mode
  /clear               - Clear conversation history
  /history             - Show conversation history
  /context-on          - Enable conversation context (default)
  /context-off         - Disable conversation context
  /stats               - Show conversation statistics and token usage

Note: Commands must start with "/" to be recognized.
      All other text is sent to the AI as prompts.
      Examples:
      - "/stats" will show statistics
      - "show me stats" will be sent to the AI as a prompt

You can also use command-line arguments:
  --model <model_id>        - Override default model
  --temperature <value>     - Set temperature (0.0-1.0)
  --top-p <value>           - Set top-p (0.0-1.0)
  --max-tokens <value>      - Set max tokens
  --debug                   - Enable debug logging
  --stream                  - Enable streaming responses
                        """)
                        continue
                    
                    if prompt_clean == '/models':
                        try:
                            print("\nFetching available Gen AI models (on-demand only)...")
                            models = client.list_gen_ai_models(provider=args.provider, on_demand_only=True)
                            print(client.format_models_list(models, show_on_demand=True))
                        except Exception as e:
                            print(f"Error listing models: {e}\n")
                        continue
                    
                    # Detect if user typed a model ID as prompt (starts with provider.model format)
                    if '.' in prompt and not ' ' in prompt and len(prompt.split('.')) >= 2:
                        # Check if it looks like a model ID
                        parts = prompt.split('.')
                        if len(parts) >= 2 and parts[0] in ['anthropic', 'amazon', 'meta', 'mistral', 'cohere', 'ai21']:
                            print(f"\n⚠️  It looks like you entered a model ID: {prompt}")
                            print(f"   To switch models, use: /use-model {prompt}")
                            print(f"   Or continue to use it as a prompt.\n")
                            # Ask for confirmation or just continue
                            continue
                    
                    if prompt_clean.startswith('/use-model '):
                        new_model = prompt[11:].strip()
                        if new_model:
                            # Show session summary before switching models
                            if session_totals['total_tokens'] > 0:
                                print(format_session_summary(session_totals, current_model))
                            # Reset session totals for new model
                            session_totals = {
                                'input_tokens': 0,
                                'output_tokens': 0,
                                'total_tokens': 0,
                                'total_cost': 0.0
                            }
                            # Clear conversation history when switching models
                            client.clear_conversation()
                            current_model = new_model
                            print(f"\n✓ Switched to model: {current_model}")
                            print("✓ Conversation history cleared\n")
                        else:
                            print("\nError: Please provide a model ID. Example: /use-model anthropic.claude-3-haiku-20240307-v1:0\n")
                        continue
                    
                    if prompt_clean == '/current-model':
                        print(f"\nCurrent model: {current_model}\n")
                        continue
                    
                    if prompt_clean == '/clear':
                        history_count = client.clear_conversation()
                        if history_count > 0:
                            print(f"\n✓ Conversation history cleared ({history_count} messages removed)\n")
                        else:
                            print("\n✓ Conversation history cleared (was already empty)\n")
                        continue
                    
                    if prompt_clean == '/history':
                        history = client.get_conversation_history()
                        if history:
                            print("\n" + "=" * 70)
                            print(f"Conversation History ({len(history)} messages)")
                            print("=" * 70)
                            for i, msg in enumerate(history, 1):
                                role = "User" if msg["role"] == "user" else "Assistant"
                                content = msg["content"]
                                # Truncate long messages for display
                                if len(content) > 200:
                                    content = content[:200] + "..."
                                print(f"{i}. [{role}]: {content}")
                            print("=" * 70)
                            print(f"Total messages in context: {len(history)}")
                            print("=" * 70 + "\n")
                        else:
                            print("\nNo conversation history.\n")
                        continue
                    
                    if prompt_clean == '/stats':
                        stats = client.conversation_manager.get_stats()
                        print("\n" + "=" * 70)
                        print("Conversation Statistics")
                        print("=" * 70)
                        print(f"Total messages: {stats['total_messages']}")
                        print(f"  - User messages: {stats['user_messages']}")
                        print(f"  - Assistant messages: {stats['assistant_messages']}")
                        print(f"  - System messages: {stats['system_messages']}")
                        print(f"\nEstimated tokens in history: {stats['estimated_tokens']:,}")
                        if stats['max_tokens']:
                            print(f"Max tokens limit: {stats['max_tokens']:,}")
                            usage_pct = (stats['estimated_tokens'] / stats['max_tokens']) * 100
                            print(f"Token usage: {usage_pct:.1f}%")
                        if stats['max_history']:
                            print(f"Max messages limit: {stats['max_history']}")
                        print(f"Trim strategy: {stats['trim_strategy']}")
                        print("=" * 70 + "\n")
                        continue
                    
                    if prompt_clean in ['/context-on', '/context-off']:
                        use_context = prompt_clean == '/context-on'
                        print(f"\n✓ Conversation context: {'ON' if use_context else 'OFF'}\n")
                        continue
                    
                    if prompt_clean in ['/stream', '/stream-on', '/stream-off']:
                        if prompt_clean == '/stream':
                            streaming_mode = not streaming_mode
                        elif prompt_clean == '/stream-on':
                            streaming_mode = True
                        else:
                            streaming_mode = False
                        print(f"\n✓ Streaming mode: {'ON' if streaming_mode else 'OFF'}\n")
                        continue
                    
                    # If we get here, it's a regular prompt - send to AI
                    # All commands must be exact matches, so any other text goes to the model
                    
                    # Estimate and show input tokens immediately (for user awareness)
                    estimated_input_tokens = estimate_input_tokens(prompt)
                    print(f"\n[Input tokens (estimated): {estimated_input_tokens}]")
                    
                    if streaming_mode:
                        # Streaming mode
                        print("Processing (streaming)...\n")
                        print("=" * 70)
                        print("Response:")
                        print("=" * 70)
                        
                        response_text = ""
                        usage = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
                        
                        try:
                            for event in client.get_response_stream(
                                prompt=prompt,
                                model_id=current_model,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                max_tokens=args.max_tokens,
                                debug=args.debug,
                                use_history=use_context
                            ):
                                if 'chunk' in event:
                                    # Print chunk immediately (flush to show in real-time)
                                    chunk = event['chunk']
                                    print(chunk, end='', flush=True)
                                    response_text += chunk
                                elif 'usage' in event:
                                    usage = event['usage']
                        except Exception as e:
                            print(f"\nError during streaming: {e}")
                            continue
                        
                        print("\n" + "=" * 70)
                        
                        # Get actual token counts
                        actual_input_tokens = usage.get('input_tokens', 0)
                        output_tokens = usage.get('output_tokens', 0)
                        total_tokens = usage.get('total_tokens', 0)
                        
                        # Update session totals
                        session_totals['input_tokens'] += actual_input_tokens
                        session_totals['output_tokens'] += output_tokens
                        session_totals['total_tokens'] += total_tokens
                        
                        # Calculate and accumulate cost
                        cost = estimate_cost(actual_input_tokens, output_tokens, current_model)
                        if cost:
                            session_totals['total_cost'] += cost
                        
                        # Show clean formatted response at the end (especially useful in debug mode)
                        if response_text.strip():
                            print("\n" + "=" * 70)
                            print("Final Response:")
                            print("=" * 70)
                            print(response_text)
                            print("=" * 70)
                        
                        print(f"\n[Input tokens (actual): {actual_input_tokens}]")
                        print(f"[Output tokens: {output_tokens}]")
                        print("=" * 70 + "\n")
                    else:
                        # Non-streaming mode (original behavior)
                        print("Processing...")
                        
                        result = client.get_response(
                            prompt=prompt,
                            model_id=current_model,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            max_tokens=args.max_tokens,
                            debug=args.debug,
                            use_history=use_context
                        )
                        response_text = result['response']
                        usage = result['usage']
                        
                        # Get actual token counts from API response
                        actual_input_tokens = usage.get('input_tokens', 0)
                        output_tokens = usage.get('output_tokens', 0)
                        total_tokens = usage.get('total_tokens', 0)
                        
                        # Update session totals with ACTUAL tokens
                        session_totals['input_tokens'] += actual_input_tokens
                        session_totals['output_tokens'] += output_tokens
                        session_totals['total_tokens'] += total_tokens
                        
                        # Calculate and accumulate cost using ACTUAL tokens
                        cost = estimate_cost(actual_input_tokens, output_tokens, current_model)
                        if cost:
                            session_totals['total_cost'] += cost
                        
                        print("\n" + "=" * 70)
                        print("Response:")
                        print("=" * 70)
                        print(response_text)
                        print("=" * 70)
                        print(f"\n[Input tokens (actual): {actual_input_tokens}]")
                        print(f"[Output tokens: {output_tokens}]")
                        print("=" * 70 + "\n")
                    
                except KeyboardInterrupt:
                    print("\n\nInterrupted. Type 'exit' to quit or continue with another prompt.")
                    continue
                except EOFError:
                    # Show session summary before exiting
                    if session_totals['total_tokens'] > 0:
                        print(format_session_summary(session_totals, current_model))
                    print("\n\nGoodbye!")
                    break
                except Exception as e:
                    print(f"\nError: {e}\n")
                    continue
                    
        except KeyboardInterrupt:
            # Show session summary before exiting
            if session_totals['total_tokens'] > 0:
                print(format_session_summary(session_totals, current_model))
            print("\n\nExiting...")
            sys.exit(0)


if __name__ == "__main__":
    main()
