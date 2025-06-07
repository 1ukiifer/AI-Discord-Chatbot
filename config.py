"""
Configuration module for Discord AI Bot
Handles all environment variables and validation
"""

import os
import sys
from dotenv import load_dotenv
from colorama import Fore, init

# Initialize colorama
init(autoreset=True)

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the Discord bot"""
    DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
    CLIENT_ID = os.getenv('CLIENT_ID')

    # Fix creator ID handling
    _creator_id_env = os.getenv('CREATOR_ID')
    CREATOR_ID = int(_creator_id_env) if _creator_id_env and _creator_id_env.strip() else None

    # Add validation method
    @classmethod
    def validate(cls):
        """Validate configuration"""
        if not cls.DISCORD_TOKEN:
            raise ValueError("DISCORD_TOKEN is required")

        if cls.CREATOR_ID:
            print(f"Creator ID configured: {cls.CREATOR_ID}")
        else:
            print("Warning: CREATOR_ID not configured - creator features disabled")
    
    # Channel message history configuration
    MAX_CHANNEL_MESSAGES = int(os.getenv('MAX_CHANNEL_MESSAGES', 10))

    # Memory configuration
    ENABLE_MEMORY = os.getenv('ENABLE_MEMORY', 'true').lower() == 'true'
    MEMORY_FILE = os.getenv('MEMORY_FILE', 'conversation_memory.json')
    MAX_MESSAGES_PER_USER = int(os.getenv('MAX_MESSAGES_PER_USER', '50'))
    MAX_CONVERSATION_HISTORY = os.getenv('MAX_CONVERSATION_HISTORY', 'true').lower() == 'true'
    MEMORY_CLEANUP_DAYS = int(os.getenv('MEMORY_CLEANUP_DAYS', '7'))

    # AI Service configuration
    AI_PROVIDER = os.getenv('AI_PROVIDER', 'gemini').lower()
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

    # Bot behavior
    BOT_PREFIX = os.getenv('BOT_PREFIX', '!ai')
    MAX_MESSAGE_LENGTH = int(os.getenv('MAX_MESSAGE_LENGTH', '2000'))
    RATE_LIMIT_WINDOW = int(os.getenv('RATE_LIMIT_WINDOW', '60'))
    RATE_LIMIT_MAX_REQUESTS = int(os.getenv('RATE_LIMIT_MAX_REQUESTS', '5'))

    # AI model configuration
    # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o')
    # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
    ANTHROPIC_MODEL = os.getenv('ANTHROPIC_MODEL', 'claude-3-5-sonnet-20241022')
    GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-exp')
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', '500'))
    TEMPERATURE = float(os.getenv('TEMPERATURE', '0.7'))

    # Bot triggers
    RESPOND_TO_MENTIONS = os.getenv('RESPOND_TO_MENTIONS', 'true').lower() == 'true'
    RESPOND_TO_DMS = os.getenv('RESPOND_TO_DMS', 'true').lower() == 'true'

    # Channel configuration
    ALLOWED_CHANNELS = [ch.strip() for ch in os.getenv('ALLOWED_CHANNELS', '').split(',') if ch.strip()]
    IGNORED_CHANNELS = [ch.strip() for ch in os.getenv('IGNORED_CHANNELS', '').split(',') if ch.strip()]

    # System prompt and personality
    SYSTEM_PROMPT = os.getenv('SYSTEM_PROMPT', 
        'You are a helpful and engaging AI assistant in a Discord server. You have a friendly, '
        'knowledgeable personality and adapt your communication style to match the conversation context. '
        'Be concise but informative, use appropriate Discord culture when relevant, and maintain '
        'consistency in your responses. Keep responses under 2000 characters unless detailed explanations are needed.')

    # Personality settings
    PERSONALITY_MODE = os.getenv('PERSONALITY_MODE', 'adaptive').lower()  # adaptive, professional, casual, technical
    ENABLE_EMOJI_REACTIONS = os.getenv('ENABLE_EMOJI_REACTIONS', 'false').lower() == 'true'

    # Web search configuration
    ENABLE_WEB_SEARCH = os.getenv('ENABLE_WEB_SEARCH', 'false').lower() == 'true'
    SEARCH_ENGINE = os.getenv('SEARCH_ENGINE', 'google').lower()
    MAX_SEARCH_RESULTS = int(os.getenv('MAX_SEARCH_RESULTS', '3'))

    # Search Engine API Keys
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    GOOGLE_SEARCH_ENGINE_ID = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
    BING_API_KEY = os.getenv('BING_API_KEY')
    NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')
    SERPER_API_KEY = os.getenv('SERPER_API_KEY')
    SERPAPI_KEY = os.getenv('SERPAPI_KEY')

    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()

    @classmethod
    def validate(cls):
        """Validate required configuration"""
        errors = []

        if not cls.DISCORD_TOKEN:
            errors.append("DISCORD_TOKEN is required")

        # Validate AI provider configuration
        if cls.AI_PROVIDER == 'openai' and not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required when using OpenAI provider")
        elif cls.AI_PROVIDER == 'anthropic' and not cls.ANTHROPIC_API_KEY:
            errors.append("ANTHROPIC_API_KEY is required when using Anthropic provider")
        elif cls.AI_PROVIDER == 'gemini' and not cls.GEMINI_API_KEY:
            errors.append("GEMINI_API_KEY is required when using Gemini provider")

        if errors:
            print(f"{Fore.RED}Configuration Errors:")
            for error in errors:
                print(f"{Fore.RED}  - {error}")
            sys.exit(1)

        print(f"{Fore.GREEN}✓ Configuration validated successfully")
        print(f"{Fore.CYAN}  - AI Provider: {cls.AI_PROVIDER}")
        print(f"{Fore.CYAN}  - Web Search: {'Enabled' if cls.ENABLE_WEB_SEARCH else 'Disabled'}")

