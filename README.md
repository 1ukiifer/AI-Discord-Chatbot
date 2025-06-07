# Discord AI Bot

A comprehensive Discord bot with multi-provider AI support, conversation memory, and web search capabilities.

## Features

- **Multi-Provider AI Support**: Works with OpenAI GPT, Anthropic Claude, and Google Gemini
- **Conversation Memory**: Remembers past conversations and user preferences
- **Web Search Integration**: Can search the web using multiple search engines
- **Rate Limiting**: Prevents spam and abuse
- **Channel Management**: Configurable allowed/ignored channels
- **Graceful Shutdown**: Proper cleanup and data saving
- **Comprehensive Logging**: File and console logging with configurable levels

## Quick Start

1. **Clone and Setup**
   ```bash
   git clone <repository>
   cd discord-ai-bot
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Required Configuration**
   - `DISCORD_TOKEN`: Your Discord bot token
   - Choose AI provider and set corresponding API key:
     - `OPENAI_API_KEY` for OpenAI
     - `ANTHROPIC_API_KEY` for Anthropic
     - `GEMINI_API_KEY` for Google Gemini

4. **Run the Bot**
   ```bash
   python main.py
   ```

## Configuration

### Required Settings

| Variable | Description |
|----------|-------------|
| `DISCORD_TOKEN` | Discord bot token from Discord Developer Portal |
| `AI_PROVIDER` | AI provider to use (`openai`, `anthropic`, or `gemini`) |
| `OPENAI_API_KEY` | OpenAI API key (if using OpenAI) |
| `ANTHROPIC_API_KEY` | Anthropic API key (if using Claude) |
| `GEMINI_API_KEY` | Google AI API key (if using Gemini) |

### Optional Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `BOT_PREFIX` | `!ai` | Command prefix for the bot |
| `MAX_MESSAGE_LENGTH` | `2000` | Maximum message length |
| `RESPOND_TO_MENTIONS` | `true` | Respond when mentioned |
| `RESPOND_TO_DMS` | `true` | Respond to direct messages |
| `MAX_TOKENS` | `500` | Maximum AI response tokens |
| `TEMPERATURE` | `0.7` | AI response creativity (0.0-1.0) |

### Memory Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_FILE` | `conversation_memory.json` | Memory storage file |
| `MAX_MESSAGES_PER_USER` | `50` | Messages to remember per user |
| `MEMORY_CLEANUP_DAYS` | `7` | Days before cleaning old conversations |

### Web Search Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_WEB_SEARCH` | `false` | Enable web search functionality |
| `SEARCH_ENGINE` | `google` | Preferred search engine |
| `MAX_SEARCH_RESULTS` | `3` | Maximum search results to return |

## Commands

| Command | Description |
|---------|-------------|
| `!ai <message>` | Send a message to the AI |
| `!ai status` | Show bot status and configuration |
| `!ai memory` | Show memory statistics |
| `!ai memory @user` | Show memory for specific user |
| `!ai clear_memory` | Clear your conversation memory |
| `!ai search <query>` | Perform web search (if enabled) |

## Usage Examples

### Basic Chat
