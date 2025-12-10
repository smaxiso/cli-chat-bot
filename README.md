# AI Chatbot with Text-to-Speech

A powerful CLI-based AI chatbot application with support for multiple AI providers, conversation management, streaming responses, and text-to-speech capabilities.

## Features

- 🤖 **Multiple AI Providers**: Support for Ollama, AWS Bedrock (Claude), OpenAI, Anthropic Claude, Google Gemini, and more
- 💬 **Conversation Management**: Create, switch, and manage multiple isolated conversations
- ⚡ **Response Caching**: Local caching of model responses to speed up repeated queries and save costs
- 🔄 **Streaming Responses**: Real-time streaming output for better user experience
- 🔊 **Text-to-Speech**: Optional TTS support using `pyttsx3`
- 💾 **Persistent Conversations**: Auto-save conversations to disk with isolated memory per chat
- 📊 **Token Tracking**: Track token usage, cache statistics, and estimated costs
- 🎯 **Context Preservation**: Maintain conversation context when switching between APIs

## Installation

1. Clone or navigate to the project directory:

## Commands

| Command | Description |
|---------|-------------|
| `/switch` | Switch AI provider at runtime |
| `/new-chat [name]` | Start a new conversation |
| `/switch-chat <name>` | Switch to a saved conversation |
| `/list-chats` | List all saved conversations |
| `/rename-chat <name>` | Rename current conversation |
| `/clear` | Clear current conversation history |
| `/stats` | Show conversation tokens and cache statistics |
| `/cache-clear` | Clear the response cache |
| `/tts-on` / `/tts-off` | Toggle Text-to-Speech |
| `/stream-on` / `/stream-off` | Toggle streaming |
| `/help` | Show all available commands |
| `/exit` | Exit the application |

## WSL Setup (Ollama)

If you are running Ollama on Windows and this CLI on WSL2, you might face connection issues with `localhost`.
To fix this, create a `.env` file in the project root and add your Windows Host IP:

```env
OLLAMA_BASE_URL=http://<YOUR_WINDOWS_IP>:11434
```

The application is configured to prioritize this `.env` setting over other configurations to ensure connectivity works in WSL environments.
