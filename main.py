import os
import discord
from discord.ext import commands
import logging
import asyncio
import signal
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional
import logging.handlers
import sys
import aiohttp
from colorama import Fore, Style, init
import google.generativeai as genai
import openai
import anthropic
from dotenv import load_dotenv
from media_handler import MediaHandler
import re
from memory import AIServiceWithMemory


# Initialize colorama
init(autoreset=True)

# Load environment variables
load_dotenv()

from config import Config
from ai_service import AIService
from search_service import SearchService
from memory import ConversationMemory, AIServiceWithMemory
from emoji_reactions import EmojiReactionService
from media_service import MediaService

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Validate configuration
Config.validate()

# Setup logging with rotation
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL.upper(), 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler(
            'logs/bot.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        ),
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Discord logging filter to prevent spam
class DiscordLogFilter(logging.Filter):
    """Filters out noisy Discord heartbeat logs and fixes None shard_id issues."""
    def filter(self, record):
        # Skip heartbeat messages
        if 'heartbeat' in str(record.msg).lower():
            return False

        # Fix None shard_id issues in log arguments if present
        if hasattr(record, 'args') and record.args:
            fixed_args = []
            for arg in record.args:
                if arg is None and 'shard' in str(record.msg).lower():
                    fixed_args.append(0)
                else:
                    fixed_args.append(arg)
            record.args = tuple(fixed_args)
        return True

# Apply filters to discord loggers
discord_logger = logging.getLogger('discord')
discord_logger.addFilter(DiscordLogFilter())
discord_logger.setLevel(logging.WARNING)

# Suppress overly noisy loggers
logging.getLogger('discord.gateway').setLevel(logging.ERROR)
logging.getLogger('discord.client').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('aiohttp').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)

# Setup Discord bot
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.guild_messages = True

bot = commands.Bot(command_prefix=Config.BOT_PREFIX, intents=intents)

# Global services
ai_service: Optional[AIService] = None
search_service: Optional[SearchService] = None
memory_service: Optional[ConversationMemory] = None
ai_with_memory: Optional[AIServiceWithMemory] = None
emoji_service: Optional[EmojiReactionService] = None
media_service: Optional[MediaService] = None

# Global state
active_conversations = set()
rate_limit_tracker = defaultdict(list)

def is_rate_limited(user_id: str) -> bool:
    """Check if user is rate limited based on Config settings."""
    now = datetime.now()
    user_requests = rate_limit_tracker[user_id]

    # Remove old requests outside the rate limit window
    user_requests[:] = [req_time for req_time in user_requests
                       if (now - req_time).seconds < Config.RATE_LIMIT_WINDOW]

    # Check if limit exceeded
    if len(user_requests) >= Config.RATE_LIMIT_MAX_REQUESTS:
        return True

    # Add current request timestamp
    user_requests.append(now)
    return Fals

CREATOR_ID = 123456789012345678 

def should_respond_to_message(message: discord.Message) -> bool:
    """Determine if the bot should respond to a given message."""
    # Don't respond to self
    if message.author == bot.user:
        return False

    # Don't respond to bots
    if message.author.bot:
        return False

    # Check channel restrictions
    if Config.ALLOWED_CHANNELS and str(message.channel.id) not in Config.ALLOWED_CHANNELS:
        logger.debug(f"Ignoring message in disallowed channel: {message.channel.id}")
        return False

    if Config.IGNORED_CHANNELS and str(message.channel.id) in Config.IGNORED_CHANNELS:
        logger.debug(f"Ignoring message in explicitly ignored channel: {message.channel.id}")
        return False

    # Check if it's a DM
    if isinstance(message.channel, discord.DMChannel):
        return Config.RESPOND_TO_DMS

    # Check if bot is mentioned or message starts with bot prefix OR !ai
    if Config.RESPOND_TO_MENTIONS and bot.user in message.mentions:
        return True

    if message.content.startswith(Config.BOT_PREFIX):
        return True

    # Add support for !ai command
    if message.content.startswith('!ai'):
        return True

    return False

@bot.event
async def on_message(message: discord.Message):
    """Handle incoming messages, processing AI responses, rate limits, and commands."""
    try:
        # Skip if message is from the bot itself
        if message.author == bot.user:
            return

        # Process commands first
        await bot.process_commands(message)

        # Handle media commands first - if handled, don't process further
        if await handle_media_commands(message, your_ai_client):
            return

        # Process media commands (second check) - if handled, don't process further
        if await process_media_commands(message, your_ai_client):
            return

        # Add message to channel history regardless of whether bot will respond
        if memory_service and not message.author.bot and hasattr(message.channel, 'id'):
            memory_service.add_channel_message(
                str(message.author.id),       # user_id (first parameter)
                message.author.display_name,  # username (second parameter)  
                message.content,              # content (third parameter)
                str(message.channel.id)       # channel_id (fourth parameter)
            )

        # Check if bot should respond to this message
        if not should_respond_to_message(message):
            return

        user_id = str(message.author.id)

        # Check if message is from the creator
        creator_id = getattr(Config, 'CREATOR_ID', None)
        is_creator = False
        if creator_id is not None:
            is_creator = message.author.id == creator_id
            if is_creator:
                logger.info(f"Message from CREATOR detected: {message.author.display_name} ({message.author.id})")

        # Apply rate limiting (skip for creator)
        if not is_creator and is_rate_limited(user_id):
            await message.reply("⏰ Please slow down! You're sending messages too quickly.")
            return

        # Prevent duplicate processing
        user_key = f"{message.author.id}_{message.channel.id}"
        if user_key in active_conversations:
            return

        active_conversations.add(user_key)

        try:
            # Show typing indicator while processing
            async with message.channel.typing():
                content = message.content.strip()

                # Clean message content: remove prefix and bot mentions
                if content.startswith(Config.BOT_PREFIX):
                    content = content[len(Config.BOT_PREFIX):].strip()
                elif content.startswith('!ai'):  # Handle !ai command
                    content = content[3:].strip()  # Remove "!ai" (3 characters)

                # Remove bot mentions from content
                if bot.user in message.mentions:
                    content = content.replace(f'<@{bot.user.id}>', '').strip()
                    content = content.replace(f'<@!{bot.user.id}>', '').strip()

                # If content is empty after cleaning
                if not content:
                    if is_creator:
                        await message.reply("Hello, Creator! How can I assist you today?")
                    else:
                        await message.reply("Hi! How can I help you today?")
                    return

                # Add creator context to the prompt if it's the creator
                if is_creator:
                    content = f"[MESSAGE FROM THE CREATOR]: {content}"
                    logger.info(f"Added creator context to message: {content[:100]}...")

                # Process message with timeout, including channel context for memory
                try:
                    if hasattr(ai_with_memory, 'generate_response_with_channel_context'):
                        response = await asyncio.wait_for(
                            process_message_with_search_and_context(content, user_id, str(message.channel.id)),
                            timeout=getattr(Config, 'RESPONSE_TIMEOUT_SECONDS', 30)
                        )
                    else:
                        response = await asyncio.wait_for(
                            ai_with_memory.generate_response(content, user_id),
                            timeout=getattr(Config, 'RESPONSE_TIMEOUT_SECONDS', 30)
                        )
                except asyncio.TimeoutError:
                    logger.warning(f"Message processing timeout for user {user_id} in channel {message.channel.id}")
                    await message.reply("⏱️ Request timed out. Please try a shorter message or try again later.")
                    return
                except Exception as e:
                    logger.error(f"Error during AI response generation: {e}", exc_info=True)
                    await message.reply("Sorry, I encountered an error generating a response. Please try again.")
                    return

                # Split long responses into chunks if necessary
                if len(response) > Config.MAX_MESSAGE_LENGTH:
                    chunks = []
                    current_chunk = ""

                    for line in response.splitlines(keepends=True):
                        if len(current_chunk) + len(line) > Config.MAX_MESSAGE_LENGTH - 50:
                            if current_chunk.strip():
                                chunks.append(current_chunk.strip())
                            current_chunk = ""

                            # If single line is too long, split it
                            if len(line) > Config.MAX_MESSAGE_LENGTH - 50:
                                while len(line) > Config.MAX_MESSAGE_LENGTH - 50:
                                    chunks.append(line[:Config.MAX_MESSAGE_LENGTH - 50])
                                    line = line[Config.MAX_MESSAGE_LENGTH - 50:]

                            current_chunk = line
                        else:
                            current_chunk += line

                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())

                    # Send chunks sequentially
                    for i, chunk in enumerate(chunks):
                        if chunk.strip():
                            try:
                                if i == 0:
                                    reply_msg = await message.reply(chunk.strip())
                                else:
                                    reply_msg = await message.channel.send(chunk.strip())
                                if emoji_service:
                                    await emoji_service.add_context_reactions(reply_msg, chunk)
                            except discord.HTTPException as e:
                                logger.warning(f"Failed to send message chunk: {e} | Chunk: {chunk[:50]}...")
                                await message.channel.send("*(Failed to send a part of the response.)*")
                else:
                    # Send single response
                    try:
                        reply_msg = await message.reply(response)
                        if emoji_service:
                            await emoji_service.add_context_reactions(reply_msg, response)
                    except discord.HTTPException as e:
                        logger.warning(f"Failed to send message via reply: {e}. Attempting channel send.")
                        try:
                            await message.channel.send(response)
                        except discord.HTTPException as e_fallback:
                            logger.error(f"Failed to send message to channel as fallback: {e_fallback}. Message ID: {message.id}")
                            await message.channel.send("Sorry, I couldn't send the response to you.")
                    except Exception as e:
                        logger.error(f"Unexpected error sending response: {e}", exc_info=True)
                        await message.channel.send("An unexpected error occurred while sending the response.")

        finally:
            active_conversations.discard(user_key)

    except discord.Forbidden:
        logger.warning(f"Bot missing permissions to respond in channel {message.channel.id} of guild {message.guild.id}")
    except discord.HTTPException as e:
        logger.error(f"Discord HTTP error in message handler: {e}", exc_info=True)
        if message.channel:
            try:
                await message.channel.send("❌ There was a Discord API error while processing your message. Please try again.")
            except discord.Forbidden:
                pass
    except Exception as e:
        logger.error(f"Unexpected error handling message {message.id}: {e}", exc_info=True)
        if message.channel:
            try:
                await message.channel.send("Sorry, I encountered an unexpected error. Please try again.")
            except discord.Forbidden:
                pass

@bot.command(name='ai')
async def ai_cmd(ctx: commands.Context, *, message: str):
    """Chat with AI using !ai command"""
    user_id = str(ctx.author.id)
    creator_id = getattr(Config, 'CREATOR_ID', None)
    is_creator = ctx.author.id == creator_id if creator_id else False

    # Rate limiting (skip for creator)
    if not is_creator and is_rate_limited(user_id):
        await ctx.reply("⏰ Please slow down! You're sending messages too quickly.")
        return

    # Add creator context if needed
    if is_creator:
        message = f"[MESSAGE FROM THE CREATOR]: {message}"

    async with ctx.typing():
        try:
            response = await asyncio.wait_for(
                process_message_with_search_and_context(message, user_id, str(ctx.channel.id)),
                timeout=getattr(Config, 'RESPONSE_TIMEOUT_SECONDS', 30)
            )
            
            await ctx.reply(response)
        except asyncio.TimeoutError:
            await ctx.reply("⏱️ Request timed out. Please try again.")
        except Exception as e:
            logger.error(f"Error in ai command: {e}", exc_info=True)
            await ctx.reply("Sorry, I encountered an error. Please try again.")

def should_search_web(content: str) -> bool:
    """Determine if a message requires web search based on keywords and context."""
    if not Config.ENABLE_WEB_SEARCH:
        return False

    content_lower = content.lower().strip()

    # Explicit search requests
    explicit_search = [
        'search for', 'look up', 'find information about', 'google',
        'web search', 'search the web', 'latest news', 'current news',
        'what happened', 'breaking news', 'recent developments'
    ]

    if any(trigger in content_lower for trigger in explicit_search):
        return True

    # Time-sensitive queries
    time_sensitive = [
        'today', 'yesterday', 'this week', 'this month', 'this year',
        'currently', 'right now', 'at the moment', 'latest', 'recent',
        str(datetime.now().year), 'breaking', 'live', 'happening now'
    ]

    # Current events indicators
    current_events = [
        'news about', 'what\'s happening with', 'updates on',
        'current status of', 'latest on'
    ]

    # Only search if it's clearly about current/recent information
    if any(indicator in content_lower for indicator in time_sensitive + current_events):
        return True

    # Don't search for general knowledge, definitions, or casual conversation
    avoid_search = [
        'what is', 'what are', 'define', 'explain', 'how does',
        'can you', 'please help', 'i think', 'i feel', 'opinion',
        'tell me about', 'help me understand'
    ]

    if any(avoid in content_lower for avoid in avoid_search):
        return False

    return False

class MediaCommands:
    def __init__(self, bot, ai_client):
        self.bot = bot
        self.ai_client = ai_client
        self.media_handler = None

    async def setup(self):
        """Initialize media handler"""
        self.media_handler = MediaHandler()
        await self.media_handler.__aenter__()

    async def cleanup(self):
        """Cleanup media handler"""
        if self.media_handler:
            await self.media_handler.__aexit__(None, None, None)
            self.media_handler.cleanup()

async def process_message_with_search_and_context(content: str, user_id: str, channel_id: str) -> str:
    """Process a user message, optionally performing a web search and using channel context."""
    if ai_with_memory is None or search_service is None:
        logger.error("AI or Search service not initialized. Cannot process message.")
        return "Sorry, the AI services are not fully initialized yet. Please try again later."

    try:
        needs_search = should_search_web(content)
        logger.info(f"Search needed for '{content[:50]}...': {needs_search}")

        if needs_search:
            logger.info(f"Attempting web search for: {content}")
            search_results = await search_service.search(content, Config.MAX_SEARCH_RESULTS)
            logger.info(f"Search completed. Results count: {len(search_results) if search_results else 0}")

            if search_results:
                logger.debug(f"First search result snippet: {search_results[0]['snippet'][:100]}")

                # Enhance prompt with search results
                search_context = "Current web search results:\n"
                for i, result in enumerate(search_results, 1):
                    search_context += f"{i}. {result['title']}\n"
                    search_context += f"   Snippet: {result['snippet']}\n"
                    search_context += f"   Source: <{result['url']}>\n\n"

                enhanced_content = (
                    f"{search_context}\n"
                    f"Based on the above search results, please answer the user's question. "
                    f"If the search results do not fully answer the question, use your existing knowledge. "
                    f"User question: {content}"
                )

                logger.info(f"Enhanced content length: {len(enhanced_content)}")
                logger.debug(f"Enhanced content preview: {enhanced_content[:500]}...")

                response = await ai_with_memory.generate_response_with_channel_context(enhanced_content, user_id, channel_id)
                logger.info(f"AI response length: {len(response)}")
                logger.debug(f"AI response preview: {response[:500]}...")

                return response
            else:
                logger.warning("Search was triggered but no results returned. Proceeding with regular AI response.")

        # Regular AI response with channel context if no search needed or no results
        logger.info("Using regular AI response with channel context.")
        return await ai_with_memory.generate_response_with_channel_context(content, user_id, channel_id)

    except Exception as e:
        logger.error(f"Error processing message with search/context: {e}", exc_info=True)
        return "Sorry, I encountered an error processing your request. Please try again."

async def process_message_with_search(content: str, user_id: str) -> str:
    """Process a user message for slash commands, optionally performing a web search."""
    if ai_with_memory is None or search_service is None:
        logger.error("AI or Search service not initialized. Cannot process message.")
        return "Sorry, the AI services are not fully initialized yet. Please try again later."

    try:
        needs_search = should_search_web(content)
        logger.info(f"Search needed for '{content[:50]}...': {needs_search}")

        if needs_search:
            logger.info(f"Attempting web search for: {content}")
            search_results = await search_service.search(content, Config.MAX_SEARCH_RESULTS)
            logger.info(f"Search completed. Results count: {len(search_results) if search_results else 0}")

            if search_results:
                logger.debug(f"First search result snippet: {search_results[0]['snippet'][:100]}")

                # Enhance prompt with search results
                search_context = "Current web search results:\n"
                for i, result in enumerate(search_results, 1):
                    search_context += f"{i}. {result['title']}\n"
                    search_context += f"   Snippet: {result['snippet']}\n"
                    search_context += f"   Source: <{result['url']}>\n\n"

                enhanced_content = (
                    f"{search_context}\n"
                    f"Based on the above search results, please answer the user's question. "
                    f"If the search results do not fully answer the question, use your existing knowledge. "
                    f"User question: {content}"
                )

                logger.info(f"Enhanced content length: {len(enhanced_content)}")
                logger.debug(f"Enhanced content preview: {enhanced_content[:500]}...")

                response = await ai_with_memory.generate_response(enhanced_content, user_id)
                logger.info(f"AI response length: {len(response)}")
                logger.debug(f"AI response preview: {response[:500]}...")

                return response
            else:
                logger.warning("Search was triggered but no results returned. Proceeding with regular AI response.")

        # Regular AI response if no search needed or no results
        logger.info("Using regular AI response (no search).")
        return await ai_with_memory.generate_response(content, user_id)

    except Exception as e:
        logger.error(f"Error processing message with search: {e}", exc_info=True)
        return "Sorry, I encountered an error processing your request. Please try again."

@bot.event
async def on_ready():
    """Bot ready event, handles initial setup like slash command syncing and status."""
    try:
        guild_count = len(bot.guilds)
        print(f"{Fore.GREEN}{Style.BRIGHT}🤖 Bot ready! Logged in as {bot.user}")
        print(f"{Fore.CYAN}🧠 AI Provider: {Config.AI_PROVIDER.upper()}")
        print(f"{Fore.CYAN}🔍 Search Engine: {getattr(Config, 'SEARCH_ENGINE', 'default').upper()}")
        logger.info(f"Bot connected to {guild_count} guilds using {Config.AI_PROVIDER}")

        # Load channel message histories for context
        if memory_service:
            logger.info("Loading channel message histories...")
            await load_channel_histories()
        else:
            logger.warning("Memory service not initialized, skipping channel history loading.")

        # Sync slash commands per guild to ensure they are up-to-date
        for guild in bot.guilds:
            try:
                await bot.tree.sync(guild=guild)
                logger.info(f"Synced slash commands to guild: {guild.name} ({guild.id})")
            except discord.Forbidden:
                logger.warning(f"Missing permissions to sync commands to {guild.name} ({guild.id}).")
            except Exception as e:
                logger.error(f"Failed to sync commands to {guild.name} ({guild.id}): {e}")

        # Set bot status
        activity = discord.Activity(type=discord.ActivityType.listening, name=f"{Config.BOT_PREFIX}help | /chat")
        await bot.change_presence(activity=activity)
        logger.info("Bot status set.")

    except Exception as e:
        logger.error(f"Error in on_ready event: {e}", exc_info=True)


# Basic Commands
@bot.command(name='ping')
async def ping_cmd(ctx: commands.Context):
    """Check bot latency"""
    latency = round(bot.latency * 1000)
    await ctx.reply(f'🏓 Pong! {latency}ms')

@bot.command(name='status')
async def status_cmd(ctx: commands.Context):
    """Show bot status and configuration."""
    try:
        if ai_service is None or memory_service is None:
            await ctx.reply("Services are not fully initialized yet. Please wait a moment.")
            return

        ai_info = ai_service.get_provider_info()
        memory_stats = memory_service.get_memory_stats()

        embed = discord.Embed(title="🤖 Bot Status", color=0x00ff00)
        embed.add_field(
            name="AI Provider",
            value=f"{ai_info['provider'].title()} ({ai_info['model']})",
            inline=True
        )
        embed.add_field(
            name="Web Search",
            value="✅ Enabled" if Config.ENABLE_WEB_SEARCH else "❌ Disabled",
            inline=True
        )
        embed.add_field(
            name="Personality",
            value=getattr(Config, 'PERSONALITY_MODE', 'Default').title(),
            inline=True
        )
        embed.add_field(
            name="Memory Stats",
            value=(
                f"Users: {memory_stats['total_users']}\n"
                f"Messages: {memory_stats['total_messages']}\n"
                f"File Size: {memory_stats['memory_file_size_kb']} KB"
            ),
            inline=True
        )
        embed.add_field(
            name="Configuration",
            value=(
                f"Max Tokens: {ai_info['max_tokens']}\n"
                f"Temperature: {ai_info['temperature']}\n"
                f"Rate Limit: {Config.RATE_LIMIT_MAX_REQUESTS}/{Config.RATE_LIMIT_WINDOW}s"
            ),
            inline=False
        )
        embed.add_field(name="Guilds", value=len(bot.guilds), inline=True)
        embed.add_field(name="Active Chats", value=len(active_conversations), inline=True)

        await ctx.reply(embed=embed)

    except Exception as e:
        logger.error(f"Error in status command: {e}", exc_info=True)
        await ctx.reply("Error retrieving status information.")

@bot.command(name='memory')
async def memory_cmd(ctx: commands.Context, user: Optional[discord.Member] = None):
    """Show memory information for a specific user or overall bot memory statistics."""
    try:
        if memory_service is None:
            await ctx.reply("Memory service is not initialized yet.")
            return

        if user:
            # Show specific user's memory
            user_summary = memory_service.get_user_summary(str(user.id))

            if user_summary.get('total_messages', 0) == 0:
                await ctx.reply(f"No conversation history found for {user.display_name}.")
                return

            embed = discord.Embed(title=f"💭 Memory for {user.display_name}", color=0x0099ff)
            embed.add_field(name="Total Messages", value=user_summary['total_messages'], inline=True)
            if user_summary.get('avg_message_length') is not None:
                embed.add_field(name="Avg Message Length", value=f"{user_summary['avg_message_length']:.2f} chars", inline=True)
            else:
                embed.add_field(name="Avg Message Length", value="N/A", inline=True)

            if user_summary['interests']:
                embed.add_field(name="Interests", value=", ".join(user_summary['interests']), inline=False)

            if user_summary['recent_activity']:
                recent = "\n".join([f"• {activity['preview']}" for activity in user_summary['recent_activity'][-3:]])
                embed.add_field(name="Recent Activity", value=recent, inline=False)

            await ctx.reply(embed=embed)
        else:
            # Show overall memory stats
            stats = memory_service.get_memory_stats()
            embed = discord.Embed(title="💭 Bot Memory Statistics", color=0x0099ff)
            embed.add_field(name="Total Users", value=stats['total_users'], inline=True)
            embed.add_field(name="Total Messages", value=stats['total_messages'], inline=True)
            embed.add_field(name="Memory File Size", value=f"{stats['memory_file_size_kb']} KB", inline=True)
            embed.add_field(name="Cleanup Days", value=stats['cleanup_days'], inline=True)
            embed.add_field(name="Max Channel Messages", value=getattr(Config, 'MAX_CHANNEL_MESSAGES', 'N/A'), inline=True)

            await ctx.reply(embed=embed)

    except Exception as e:
        logger.error(f"Error in memory command: {e}", exc_info=True)
        await ctx.reply("Error retrieving memory information.")

@bot.command(name='clear_memory')
async def clear_memory_cmd(ctx: commands.Context):
    """Clear your conversation memory with the bot."""
    try:
        if memory_service is None:
            await ctx.reply("Memory service is not initialized yet.")
            return

        user_id = str(ctx.author.id)
        memory_service.clear_user_memory(user_id)
        await ctx.reply("✅ Your conversation memory has been cleared.")

    except Exception as e:
        logger.error(f"Error clearing memory for {ctx.author.id}: {e}", exc_info=True)
        await ctx.reply("Error clearing memory.")

@bot.command(name='search')
async def search_cmd(ctx: commands.Context, *, query: str):
    """Perform a web search using the bot's integrated search service."""
    try:
        if not Config.ENABLE_WEB_SEARCH:
            await ctx.reply("Web search is not enabled for this bot.")
            return
        if search_service is None:
            await ctx.reply("Search service is not initialized yet.")
            return

        # Check if user is the creator
        is_creator = ctx.author.id == getattr(Config, 'CREATOR_ID', None) if hasattr(Config, 'CREATOR_ID') else False

        # Rate limiting (skip for creator)
        if not is_creator and is_rate_limited(str(ctx.author.id)):
            await ctx.reply("⏰ Please slow down! Try again in a moment.")
            return

        async with ctx.typing():
            results = await search_service.search(query, Config.MAX_SEARCH_RESULTS)

            if not results:
                await ctx.reply("🔍 No search results found for that query.")
                return

            embed = discord.Embed(title=f"🔍 Search Results for: {query}", color=0x00ff00)

            for i, result in enumerate(results, 1):
                title = result['title']
                if len(title) > 100:
                    title = title[:97] + "..."
                snippet = result['snippet']
                if len(snippet) > 200:
                    snippet = snippet[:197] + "..."

                embed.add_field(
                    name=f"{i}. {title}",
                    value=f"{snippet}\n[View]({result['url']}) • {result['source']}",
                    inline=False
                )
                if i >= 5:
                    break

            embed.set_footer(text=f"Searched using {getattr(Config, 'SEARCH_ENGINE', 'default').title()}")
            await ctx.reply(embed=embed)

    except Exception as e:
        logger.error(f"Error in search command for query '{query}': {e}", exc_info=True)
        await ctx.reply("Error performing search. Please try again.")

@bot.command(name='switch_provider')
async def switch_provider_cmd(ctx: commands.Context, provider: str):
    """Switch the AI provider (requires available credentials)."""
    try:
        if ai_service is None:
            await ctx.reply("AI service is not initialized yet.")
            return

        available_providers = ai_service.get_available_providers()

        if provider.lower() not in available_providers:
            await ctx.reply(f"Provider '{provider}' is not available. Available providers: {', '.join(available_providers)}")
            return

        if ai_service.switch_provider(provider.lower()):
            ai_info = ai_service.get_provider_info()
            embed = discord.Embed(title="✅ Provider Switched", color=0x00ff00)
            embed.add_field(name="New Provider", value=f"{ai_info['provider'].title()}", inline=True)
            embed.add_field(name="Model", value=f"{ai_info['model']}", inline=True)
            await ctx.reply(embed=embed)
        else:
            await ctx.reply(f"Failed to switch to '{provider}'. Check if credentials for this provider are available in your configuration.")

    except Exception as e:
        logger.error(f"Error switching provider to '{provider}': {e}", exc_info=True)
        await ctx.reply("Error switching AI provider.")

@bot.command(name='personality')
async def personality_cmd(ctx: commands.Context, mode: str = None):
    """Set or view the bot's personality mode."""
    try:
        available_modes = ['adaptive', 'professional', 'casual', 'technical']

        if mode is None:
            current_mode = getattr(Config, 'PERSONALITY_MODE', 'default')
            embed = discord.Embed(title="🎭 Current Personality Mode", color=0x0099ff)
            embed.add_field(name="Mode", value=current_mode.title(), inline=True)
            embed.add_field(name="Available Modes", value=", ".join(available_modes), inline=False)
            await ctx.reply(embed=embed)
            return

        if mode.lower() not in available_modes:
            await ctx.reply(f"Invalid personality mode. Available modes are: {', '.join(available_modes)}")
            return

        # Update personality mode
        Config.PERSONALITY_MODE = mode.lower()

        embed = discord.Embed(title="✅ Personality Updated", color=0x00ff00)
        embed.add_field(name="New Mode", value=mode.title(), inline=True)

        mode_descriptions = {
            'adaptive': 'Adapts tone to match conversation context.',
            'professional': 'Maintains a formal, business-like communication style.',
            'casual': 'Uses a relaxed, friendly tone, potentially with internet culture references.',
            'technical': 'Focuses on detailed, precise, and technical explanations.'
        }

        embed.add_field(name="Description", value=mode_descriptions.get(mode.lower(), "No description available."), inline=False)
        await ctx.reply(embed=embed)

    except Exception as e:
        logger.error(f"Error setting personality mode to '{mode}': {e}", exc_info=True)
        await ctx.reply("Error updating personality mode.")

@bot.command(name='engines')
async def list_engines(ctx):
    """List available search engines and their status"""
    embed = discord.Embed(title="🔍 Available Search Engines", color=0x0099FF)

    engines = {
        'Google': bool(getattr(Config, 'GOOGLE_API_KEY', None) and getattr(Config, 'GOOGLE_SEARCH_ENGINE_ID', None)),
        'Serper': bool(getattr(Config, 'SERPER_API_KEY', None)),
        'Bing': bool(getattr(Config, 'BING_API_KEY', None)),
        'NewsAPI': bool(getattr(Config, 'NEWSAPI_KEY', None)),
        'DuckDuckGo': True,  # Always available (free)
    }

    current_engine = getattr(Config, 'SEARCH_ENGINE', 'default')

    for engine, available in engines.items():
        status = "✅ Available" if available else "❌ Not configured"
        if engine.lower() == current_engine.lower():
            status += " **(Current)**"

        embed.add_field(name=engine, value=status, inline=True)

    embed.add_field(
        name="ℹ️ Configuration Help",
        value="Add the required API keys to your .env file and restart the bot.",
        inline=False
    )

    embed.set_footer(text=f"Web Search: {'Enabled' if Config.ENABLE_WEB_SEARCH else 'Disabled'}")
    await ctx.send(embed=embed)

@bot.command(name='switch_search')
@commands.has_permissions(administrator=True)
async def switch_search_engine(ctx, engine: str):
    """Switch search engine (admin only)"""
    engine = engine.lower()
    valid_engines = ['google', 'serper', 'bing', 'newsapi', 'duckduckgo']

    if engine not in valid_engines:
        await ctx.send(f"❌ Invalid search engine. Choose from: {', '.join(valid_engines)}")
        return

    # Check if engine is configured
    engine_checks = {
        'google': bool(getattr(Config, 'GOOGLE_API_KEY', None) and getattr(Config, 'GOOGLE_SEARCH_ENGINE_ID', None)),
        'serper': bool(getattr(Config, 'SERPER_API_KEY', None)),
        'bing': bool(getattr(Config, 'BING_API_KEY', None)),
        'newsapi': bool(getattr(Config, 'NEWSAPI_KEY', None)),
        'duckduckgo': True,  # Always available
    }

    if not engine_checks.get(engine, False):
        await ctx.send(f"❌ {engine.title()} is not configured. Please add the required API keys.")
        return

    Config.SEARCH_ENGINE = engine
    await ctx.send(f"✅ Switched to **{engine.title()}** search engine.")
    logger.info(f"Search engine switched to {engine} by {ctx.author}")


@bot.command(name='help_extended')
async def help_extended(ctx):
    """Show extended help with all available commands"""
    embed = discord.Embed(
        title="🤖 Discord AI Bot - Extended Help", 
        description="Here are all available commands and features:",
        color=0x00ff00
    )

    # Basic commands
    embed.add_field(
        name="📊 **Basic Commands**",
        value=f"`{Config.BOT_PREFIX} ping` - Check bot latency\n"
              f"`{Config.BOT_PREFIX} status` - Show bot status\n"
              f"`{Config.BOT_PREFIX} creator_status` - Check creator status\n"
              f"`{Config.BOT_PREFIX} help_extended` - Show this help",
        inline=False
    )

    # Search commands
    embed.add_field(
        name="🔍 **Search Commands**",
        value=f"`{Config.BOT_PREFIX} search <query>` - Search the web\n"
              f"`{Config.BOT_PREFIX} engines` - List available search engines",
        inline=False
    )

    # Memory commands
    embed.add_field(
        name="🧠 **Memory Commands**",
        value=f"`{Config.BOT_PREFIX} memory` - Show memory statistics\n"
              f"`{Config.BOT_PREFIX} clear_memory` - Clear your conversation memory",
        inline=False
    )

    # Admin commands
    embed.add_field(
        name="⚙️ **Admin Commands**",
        value=f"`{Config.BOT_PREFIX} switch_provider <provider>` - Switch AI provider\n"
              f"`{Config.BOT_PREFIX} switch_search <engine>` - Switch search engine\n"
              f"`{Config.BOT_PREFIX} personality <mode>` - Set personality mode",
        inline=False
    )

    # AI interaction
    embed.add_field(
        name="💬 **AI Interaction**",
        value="• Mention the bot: `@BotName your message`\n"
              f"• Use prefix: `{Config.BOT_PREFIX} your message`\n"
              "• Direct message the bot\n"
              "• Bot remembers conversation context",
        inline=False
    )

    embed.set_footer(text="💡 Tip: The bot can search the web and remember conversations!")
    await ctx.send(embed=embed)

# Slash Commands
@bot.tree.command(name="chat", description="Chat with the AI assistant.")
@discord.app_commands.describe(message="Your message to the AI.")
async def chat_slash(interaction: discord.Interaction, message: str):
    """Slash command for chatting with AI."""
    try:
        # Defer immediately - within 3 seconds of receiving interaction
        await interaction.response.defer()

        user_id = str(interaction.user.id)
        is_creator = interaction.user.id == getattr(Config, 'CREATOR_ID', None) if hasattr(Config, 'CREATOR_ID') else False


        # Rate limiting (skip for creator)
        if not is_creator and is_rate_limited(user_id):
            await interaction.followup.send("⏰ Please slow down! You're sending messages too quickly.")
            return

        # Process message with timeout
        try:
            response = await asyncio.wait_for(
                process_message_with_search(message, user_id),
                timeout=25  # Leave 5 seconds buffer before Discord's 30s limit
            )
        except asyncio.TimeoutError:
            await interaction.followup.send("⏱️ Request timed out. Please try a shorter message.")
            return

        # Send response
        if len(response) > Config.MAX_MESSAGE_LENGTH:
            # Split and send chunks
            chunks = [response[i:i + Config.MAX_MESSAGE_LENGTH] for i in range(0, len(response), Config.MAX_MESSAGE_LENGTH)]
            for i, chunk in enumerate(chunks):
                if i == 0:
                    await interaction.followup.send(chunk)
                else:
                    await interaction.followup.send(chunk)  # Use followup for all chunks
        else:
            await interaction.followup.send(response)

    except discord.NotFound:
        # Interaction expired or invalid - log but don't crash
        logger.warning(f"Interaction expired for user {interaction.user.id} with message: {message[:50]}...")
        return
    except Exception as e:
        logger.error(f"Error in chat slash command: {e}", exc_info=True)
        try:
            if not interaction.response.is_done():
                await interaction.response.send_message("Sorry, I encountered an error. Please try again.", ephemeral=True)
            else:
                await interaction.followup.send("Sorry, I encountered an error. Please try again.", ephemeral=True)
        except discord.NotFound:
            # If we can't respond, just log it
            logger.warning("Could not send error message - interaction expired")
            
@bot.tree.command(name="status", description="Show bot status and configuration.")
async def status_slash(interaction: discord.Interaction):
    """Slash command for bot status."""
    try:
        await interaction.response.defer()

        if ai_service is None or memory_service is None:
            await interaction.followup.send("Services are not fully initialized yet. Please wait a moment.")
            return

        ai_info = ai_service.get_provider_info()
        memory_stats = memory_service.get_memory_stats()

        embed = discord.Embed(title="🤖 Bot Status", color=0x00ff00)
        embed.add_field(
            name="AI Provider",
            value=f"{ai_info['provider'].title()} ({ai_info['model']})",
            inline=True
        )
        embed.add_field(
            name="Web Search",
            value="✅ Enabled" if Config.ENABLE_WEB_SEARCH else "❌ Disabled",
            inline=True
        )
        embed.add_field(
            name="Personality",
            value=getattr(Config, 'PERSONALITY_MODE', 'Default').title(),
            inline=True
        )
        embed.add_field(
            name="Memory Stats",
            value=(
                f"Users: {memory_stats['total_users']}\n"
                f"Messages: {memory_stats['total_messages']}\n"
                f"File Size: {memory_stats['memory_file_size_kb']} KB"
            ),
            inline=True
        )
        embed.add_field(
            name="Configuration",
            value=(
                f"Max Tokens: {ai_info['max_tokens']}\n"
                f"Temperature: {ai_info['temperature']}\n"
                f"Rate Limit: {Config.RATE_LIMIT_MAX_REQUESTS}/{Config.RATE_LIMIT_WINDOW}s"
            ),
            inline=False
        )

        await interaction.followup.send(embed=embed)

    except Exception as e:
        logger.error(f"Error in status slash command: {e}", exc_info=True)
        await interaction.followup.send("Error retrieving status information.")

@bot.tree.command(name="switch_provider", description="Switch the AI provider (requires available credentials).")
@discord.app_commands.describe(provider="AI provider to switch to.")
@discord.app_commands.choices(provider=[
    discord.app_commands.Choice(name="OpenAI", value="openai"),
    discord.app_commands.Choice(name="Anthropic", value="anthropic"),
    discord.app_commands.Choice(name="Gemini", value="gemini")
])
async def switch_provider_slash(interaction: discord.Interaction, provider: discord.app_commands.Choice[str]):
    """Slash command to switch AI provider."""
    try:
        await interaction.response.defer()

        if ai_service is None:
            await interaction.followup.send("AI service is not initialized yet.")
            return

        available_providers = ai_service.get_available_providers()

        if provider.value not in available_providers:
            await interaction.followup.send(f"Provider '{provider.name}' is not available. Available providers: {', '.join(available_providers)}")
            return

        if ai_service.switch_provider(provider.value):
            ai_info = ai_service.get_provider_info()
            embed = discord.Embed(title="✅ Provider Switched", color=0x00ff00)
            embed.add_field(name="New Provider", value=f"{ai_info['provider'].title()}", inline=True)
            embed.add_field(name="Model", value=f"{ai_info['model']}", inline=True)
            await interaction.followup.send(embed=embed)
        else:
            await interaction.followup.send(f"Failed to switch to '{provider.name}'. Check if credentials are available.")

    except Exception as e:
        logger.error(f"Error in switch provider slash command: {e}", exc_info=True)
        await interaction.followup.send("Error switching AI provider.")

@bot.tree.command(name="personality", description="Set bot personality mode.")
@discord.app_commands.describe(mode="Personality mode to set.")
@discord.app_commands.choices(mode=[
    discord.app_commands.Choice(name="Adaptive", value="adaptive"),
    discord.app_commands.Choice(name="Professional", value="professional"),
    discord.app_commands.Choice(name="Casual", value="casual"),
    discord.app_commands.Choice(name="Technical", value="technical")
])
async def personality_slash(interaction: discord.Interaction, mode: Optional[discord.app_commands.Choice[str]] = None):
    """Slash command to set or view personality mode."""
    try:
        await interaction.response.defer()

        available_modes = ['adaptive', 'professional', 'casual', 'technical']

        if mode is None:
            current_mode = getattr(Config, 'PERSONALITY_MODE', 'default')
            embed = discord.Embed(title="🎭 Current Personality Mode", color=0x0099ff)
            embed.add_field(name="Mode", value=current_mode.title(), inline=True)
            embed.add_field(name="Available Modes", value=", ".join(available_modes), inline=False)
            await interaction.followup.send(embed=embed)
            return

        # Update personality mode
        Config.PERSONALITY_MODE = mode.value

        embed = discord.Embed(title="✅ Personality Updated", color=0x00ff00)
        embed.add_field(name="New Mode", value=mode.name, inline=True)

        mode_descriptions = {
            'adaptive': 'Adapts tone to match conversation context.',
            'professional': 'Maintains a formal, business-like communication style.',
            'casual': 'Uses a relaxed, friendly tone, potentially with internet culture references.',
            'technical': 'Focuses on detailed, precise, and technical explanations.'
        }

        embed.add_field(name="Description", value=mode_descriptions.get(mode.value, "No description available."), inline=False)
        await interaction.followup.send(embed=embed)

    except Exception as e:
        logger.error(f"Error in personality slash command: {e}", exc_info=True)
        await interaction.followup.send("Error updating personality mode.")

@bot.tree.command(name="search", description="Search the web.")
@discord.app_commands.describe(query="What to search for.")
async def search_slash(interaction: discord.Interaction, query: str):
    """Slash command for web search."""
    try:
        # Defer immediately
        await interaction.response.defer()

        if not Config.ENABLE_WEB_SEARCH:
            await interaction.followup.send("Web search is not enabled for this bot.")
            return
        if search_service is None:
            await interaction.followup.send("Search service is not initialized yet.")
            return

        # Add timeout for search
        try:
            results = await asyncio.wait_for(
                search_service.search(query, Config.MAX_SEARCH_RESULTS),
                timeout=20
            )
        except asyncio.TimeoutError:
            await interaction.followup.send("⏱️ Search timed out. Please try again.")
            return

        if not results:
            await interaction.followup.send("No search results found for that query.")
            return

        embed = discord.Embed(title=f"🔍 Search Results for: {query}", color=0x00ff00)

        for i, result in enumerate(results, 1):
            title = result['title']
            if len(title) > 100:
                title = title[:97] + "..."
            snippet = result['snippet']
            if len(snippet) > 200:
                snippet = snippet[:197] + "..."

            embed.add_field(
                name=f"{i}. {title}",
                value=f"{snippet}\n[View]({result['url']})",
                inline=False
            )
            if i >= 5:
                break

        await interaction.followup.send(embed=embed)

    except discord.NotFound:
        logger.warning(f"Search interaction expired for query: {query}")
        return
    except Exception as e:
        logger.error(f"Error in search slash command: {e}", exc_info=True)
        try:
            if not interaction.response.is_done():
                await interaction.response.send_message("Error performing search. Please try again.", ephemeral=True)
            else:
                await interaction.followup.send("Error performing search. Please try again.", ephemeral=True)
        except discord.NotFound:
            logger.warning("Could not send search error message - interaction expired")
# Error handling
@bot.event
async def on_command_error(ctx, error):
    """Handle command errors gracefully"""
    if isinstance(error, commands.CommandNotFound):
        return
    elif isinstance(error, commands.MissingPermissions):
        await ctx.send("❌ You don't have permission to use this command.")
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"❌ Missing required argument. Use `{Config.BOT_PREFIX} help` for command usage.")
    elif isinstance(error, commands.BadArgument):
        await ctx.send("❌ Invalid argument provided.")
    else:
        logger.error(f"Command error: {error}")
        await ctx.send("❌ An unexpected error occurred while processing the command.")

@bot.event
async def on_error(event, *args, **kwargs):
    """Handle general bot errors"""
    logger.error(f"Bot error in {event}: {args}, {kwargs}")

async def shutdown_handler():
    """Handles graceful shutdown of the bot and its services."""
    logger.info("Initiating graceful shutdown...")

    try:
        # Save memory
        if memory_service:
            memory_service.save_memory()
            logger.info("Conversation memory saved.")

        # Close search service
        if search_service:
            await search_service.close()
            logger.info("Search service closed.")

        # Close bot connection
        await bot.close()
        logger.info("Discord bot connection closed.")

    except Exception as e:
        logger.error(f"Error during shutdown process: {e}", exc_info=True)
    finally:
        logger.info("Shutdown complete.")

def signal_handler(signum, frame):
    """Callback for OS signals (e.g., Ctrl+C) to initiate graceful shutdown."""
    print(f"\n{Fore.YELLOW}Shutting down gracefully...")
    logger.info(f"Received shutdown signal {signum}. Preparing for shutdown...")
    
    loop = asyncio.get_event_loop()
    if loop.is_running():
        loop.create_task(shutdown_handler())
    else:
        asyncio.run(shutdown_handler())
    
    sys.exit(0)

async def load_channel_histories():
    """Loads recent message history for all accessible text channels for context."""
    if memory_service is None:
        logger.warning("Memory service not available, skipping channel history loading.")
        return

    loaded_channels_count = 0
    skipped_channels_count = 0
    logger.info("Starting to load channel message histories...")

    for guild in bot.guilds:
        for channel in guild.text_channels:
            try:
                # Check if bot has necessary permissions
                permissions = channel.permissions_for(guild.me)
                if permissions.read_message_history and permissions.read_messages:
                    if hasattr(memory_service, 'load_channel_history'):
                        await memory_service.load_channel_history(channel, limit=getattr(Config, 'MAX_CHANNEL_MESSAGES', 30))
                    loaded_channels_count += 1
                    logger.debug(f"Loaded history for channel: #{channel.name} in {guild.name}")
                    await asyncio.sleep(0.1)
                else:
                    skipped_channels_count += 1
                    logger.debug(f"Skipping channel {channel.name} due to missing permissions.")

            except discord.Forbidden:
                skipped_channels_count += 1
                logger.warning(f"Bot lacks permissions to read history in channel: #{channel.name} ({channel.id}) in guild: {guild.name}")
            except Exception as e:
                skipped_channels_count += 1
                logger.error(f"Error loading history for channel #{channel.name} ({channel.id}) in guild {guild.name}: {e}", exc_info=True)
                continue

    logger.info(f"Finished loading channel histories. Loaded {loaded_channels_count} channels, skipped {skipped_channels_count} channels.")

async def main():
    """Main function to initialize and start the Discord bot."""
    global ai_service, search_service, memory_service, ai_with_memory, emoji_service, media_service

    logger.info("Starting bot initialization process...")
    print(f"{Fore.CYAN}{Style.BRIGHT}🚀 Starting Discord AI Bot...")
    print(f"{Fore.CYAN}🧠 AI Provider: {Config.AI_PROVIDER.upper()}")
    print(f"{Fore.CYAN}🔍 Search Engine: {getattr(Config, 'SEARCH_ENGINE', 'default').upper()}")
    print(f"{Fore.CYAN}🌐 Web Search: {'Enabled' if Config.ENABLE_WEB_SEARCH else 'Disabled'}")
    print(f"{Fore.CYAN}👑 Creator ID: {getattr(Config, 'CREATOR_ID', 'Not set') if hasattr(Config, 'CREATOR_ID') else 'Not set'}")

    try:
        # Initialize services with robust error handling
        try:
            ai_service = AIService()
            logger.info("AI service initialized successfully.")
        except Exception as e:
            logger.critical(f"Failed to initialize AI service. Please check your configuration and API keys: {e}")
            raise

        try:
            search_service = SearchService()
            logger.info("Search service initialized successfully.")
        except Exception as e:
            logger.critical(f"Failed to initialize Search service. Web search functionality will be unavailable: {e}")
            search_service = None

        try:
            max_channel_messages_setting = getattr(Config, 'MAX_CHANNEL_MESSAGES', 30)
            memory_service = ConversationMemory(
                memory_file=Config.MEMORY_FILE,
                max_messages_per_user=Config.MAX_MESSAGES_PER_USER,
                cleanup_days=Config.MEMORY_CLEANUP_DAYS,
            )
            logger.info("Memory service initialized successfully.")
        except Exception as e:
            logger.critical(f"Failed to initialize Memory service. Conversation memory will not function: {e}")
            raise

        try:
            ai_with_memory = AIServiceWithMemory(memory_service, ai_service)
            logger.info("AI with memory service initialized successfully.")
        except Exception as e:
            logger.critical(f"Failed to initialize AI with memory wrapper: {e}")
            raise

        try:
            emoji_service = EmojiReactionService()
            logger.info("Emoji reaction service initialized.")
        except Exception as e:
            logger.warning(f"Failed to initialize emoji reaction service. Bot will function without emoji reactions: {e}")
            emoji_service = None

        try:
            media_service = MediaService(max_file_size_mb=25, cache_dir="media_cache")
            logger.info("Media service initialized.")
        except Exception as e:
            logger.warning(f"Failed to initialize media service. Bot will function without media processing: {e}")
            media_service = None

        logger.info("All essential services initialized successfully.")

        # Setup OS signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start the Discord bot with connection retry logic
        max_retries = 5
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to start Discord bot (attempt {attempt + 1}/{max_retries})...")
                await bot.start(Config.DISCORD_TOKEN)
                logger.info("Discord bot started successfully.")
                break
            except discord.LoginFailure:
                print(f"{Fore.RED}❌ Invalid Discord token. Please check your DISCORD_TOKEN.")
                logger.critical("Invalid Discord token provided! Please check your DISCORD_TOKEN in config.")
                raise
            except discord.HTTPException as e:
                logger.error(f"Discord HTTP error during bot start: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying Discord connection in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.critical("Failed to connect to Discord after multiple retries. Exiting.")
                    raise
            except Exception as e:
                logger.critical(f"An unexpected error occurred while starting the bot: {e}", exc_info=True)
                if attempt < max_retries - 1:
                    logger.info(f"Retrying bot start in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.critical("Failed to start bot after multiple unexpected errors. Exiting.")
                    raise

    except Exception as e:
        logger.critical(f"Bot startup failed due to a critical error: {e}", exc_info=True)
        print(f"{Fore.RED}❌ Bot failed to start: {e}")
        await shutdown_handler()
        raise

if __name__ == "__main__":
    try:
        # Run the main asynchronous function
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Bot stopped by user.")
        logger.info("Bot process interrupted by user (KeyboardInterrupt).")
    except Exception as e:
        print(f"{Fore.RED}Fatal error: {e}")
        logger.critical(f"An unhandled exception occurred outside of main execution: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Bot application finished.")


async def handle_image_command(message, args, ai_client, media_handler):
    """Handle !image command to analyze images"""
    try:
        # Check for attachments
        image_urls = []

        # Get URLs from attachments
        for attachment in message.attachments:
            if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']):
                image_urls.append(attachment.url)

        # Get URLs from command arguments
        if args:
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            found_urls = re.findall(url_pattern, ' '.join(args))
            image_urls.extend(found_urls)

        if not image_urls:
            await message.reply("❌ Please provide an image attachment or image URL to analyze.")
            return

        await message.add_reaction("🔍")  # Processing reaction

        results = []
        for url in image_urls[:3]:  # Limit to 3 images
            async with MediaHandler() as handler:
                result, error = await handler.process_media(url)
                if error:
                    results.append(f"❌ Error processing {url}: {error}")
                else:
                    # Prepare analysis for AI
                    analysis_prompt = f"""
Analyze this image data:
- Format: {result['format']}
- Size: {result['size']}
- Text found: {result['text']}

Please provide insights about what you can determine from this image based on the extracted text and metadata.
"""

                    # Get AI analysis
                    ai_response = await ai_client.get_response(analysis_prompt, message.author.id)
                    results.append(f"🖼️ **Image Analysis:**\n{ai_response}")

        # Send results
        for result in results:
            await message.reply(result[:2000])  # Discord message limit

    except Exception as e:
        await message.reply(f"❌ Error analyzing image: {str(e)}")

async def handle_file_command(message, args, ai_client, media_handler):
    """Handle !file command to analyze documents"""
    try:
        file_urls = []

        # Get URLs from attachments
        for attachment in message.attachments:
            file_urls.append(attachment.url)

        # Get URLs from command arguments
        if args:
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            found_urls = re.findall(url_pattern, ' '.join(args))
            file_urls.extend(found_urls)

        if not file_urls:
            await message.reply("❌ Please provide a file attachment or file URL to analyze.")
            return

        await message.add_reaction("📄")  # Processing reaction

        results = []
        for url in file_urls[:2]:  # Limit to 2 files
            async with MediaHandler() as handler:
                result, error = await handler.process_media(url)
                if error:
                    results.append(f"❌ Error processing {url}: {error}")
                else:
                    # Prepare analysis for AI
                    analysis_prompt = f"""
Analyze this file data:
- Type: {result['type']}
- Size: {result['size']} bytes
- Content: {result['content']}

Please provide insights about this file and summarize its content if it's readable text.
"""

                    # Get AI analysis
                    ai_response = await ai_client.get_response(analysis_prompt, message.author.id)
                    results.append(f"📄 **File Analysis:**\n{ai_response}")

        # Send results
        for result in results:
            await message.reply(result[:2000])  # Discord message limit

    except Exception as e:
        await message.reply(f"❌ Error analyzing file: {str(e)}")

async def handle_link_command(message, args, ai_client, media_handler):
    """Handle !link command to scrape and analyze web pages"""
    try:
        if not args:
            await message.reply("❌ Please provide a URL to analyze. Usage: `!link <URL>`")
            return

        url = args[0]
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        async with MediaHandler() as handler:
            if not handler.is_valid_url(url):
                await message.reply("❌ Invalid URL provided.")
                return

            await message.add_reaction("🔗")  # Processing reaction

            result, error = await handler.scrape_webpage(url)
            if error:
                await message.reply(f"❌ Error scraping webpage: {error}")
                return

            # Prepare analysis for AI
            analysis_prompt = f"""
Analyze this webpage data:
- Title: {result['title']}
- Description: {result['description']}
- URL: {result['url']}
- Content: {result['content']}

Please provide a summary of this webpage and highlight the key information found.
"""

            # Get AI analysis
            ai_response = await ai_client.get_response(analysis_prompt, message.author.id)

            response = f"🔗 **Webpage Analysis:**\n**Title:** {result['title']}\n**URL:** {result['url']}\n\n{ai_response}"
            await message.reply(response[:2000])  # Discord message limit

    except Exception as e:
        await message.reply(f"❌ Error analyzing webpage: {str(e)}")

async def handle_search_command(message, args, ai_client):
    """Enhanced search command with AI analysis"""
    try:
        if not args:
            await message.reply("❌ Please provide search terms. Usage: `!search <query>`")
            return

        query = ' '.join(args)
        await message.add_reaction("🔍")  # Processing reaction

        # Use your existing web search functionality
        # This assumes you have a search function in your bot
        search_results = await perform_web_search(query)  # You'll need to implement this based on your existing search

        if not search_results:
            await message.reply("❌ No search results found.")
            return

        # Prepare search results for AI analysis
        search_summary = f"Search query: {query}\n\nResults:\n"
        for i, result in enumerate(search_results[:5], 1):
            search_summary += f"{i}. {result.get('title', 'No title')}\n{result.get('snippet', 'No description')}\n\n"

        analysis_prompt = f"""
Based on these search results, provide a comprehensive answer to the query: "{query}"

{search_summary}

Please synthesize the information and provide a helpful response.
"""

        # Get AI analysis
        ai_response = await ai_client.get_response(analysis_prompt, message.author.id)

        response = f"🔍 **Search Results for:** {query}\n\n{ai_response}"
        await message.reply(response[:2000])  # Discord message limit

    except Exception as e:
        await message.reply(f"❌ Error performing search: {str(e)}")

# Add this to your main message handler (where you process commands)
async def process_media_commands(message, ai_client):
    """Process media-related commands with ! prefix"""
    content = message.content.strip()

    if not content.startswith('!'):
        return False

    # Remove prefix and split command
    command_part = content[1:].strip()
    parts = command_part.split()

    if not parts:
        return False

    command = parts[0].lower()
    args = parts[1:]

    # Initialize media handler
    media_handler = MediaHandler()

    try:
        if command == 'image':
            await handle_image_command(message, args, ai_client, media_handler)
            return True
        elif command == 'file':
            await handle_file_command(message, args, ai_client, media_handler)
            return True
        elif command == 'link':
            await handle_link_command(message, args, ai_client, media_handler)
            return True
        elif command == 'search':
            await handle_search_command(message, args, ai_client)
            return True
        elif command == 'help':
            help_text = """
🤖 **Media Analysis Commands:**

`!image` - Analyze images (attach image or provide URL)
`!file` - Analyze documents and files (attach file or provide URL)
`!link <URL>` - Scrape and analyze web pages
`!search <query>` - Enhanced web search with AI analysis
`!help` - Show this help message

**Examples:**
• `!image` (with image attachment)
• `!file` (with document attachment)
• `!link https://example.com`
• `!search artificial intelligence news`
"""
            await message.reply(help_text)
            return True
    except Exception as e:
        await message.reply(f"❌ Command error: {str(e)}")
    finally:
        media_handler.cleanup()

    return False


async def handle_media_commands(message, ai_client):
    """Handle media commands with ! prefix"""
    content = message.content.strip()

    # Check if it's a command
    if not content.startswith('!'):
        return False

    # Parse command
    parts = content[1:].split()
    if not parts:
        return False

    command = parts[0].lower()
    args = parts[1:]

    print(f"Processing command: {command}")  # Debug line

    try:
        if command == 'help':
            help_text = """
🤖 **Media Analysis Commands:**

`!image` - Analyze images (attach image or provide URL)
`!file` - Analyze documents and files (attach file or provide URL)
`!link <URL>` - Scrape and analyze web pages
`!search <query>` - Enhanced web search with AI analysis
`!help` - Show this help message

**Examples:**
• `!image` (with image attachment)
• `!file` (with document attachment)
• `!link https://example.com`
• `!search artificial intelligence news`
"""
            await message.reply(help_text)
            return True

        elif command == 'image':
            await handle_image_analysis(message, args, ai_client)
            return True

        elif command == 'file':
            await handle_file_analysis(message, args, ai_client)
            return True

        elif command == 'link':
            await handle_link_analysis(message, args, ai_client)
            return True

        elif command == 'search':
            await handle_search_analysis(message, args, ai_client)
            return True

    except Exception as e:
        await message.reply(f"❌ Error processing command: {str(e)}")
        print(f"Command error: {e}")  # Debug line
        return True

    return False

async def handle_image_analysis(message, args, ai_client):
    """Handle image analysis"""
    print("Handling image analysis...")  # Debug line

    # Get image URLs
    image_urls = []

    # Check attachments
    for attachment in message.attachments:
        if attachment.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
            image_urls.append(attachment.url)
            print(f"Found image attachment: {attachment.url}")  # Debug line

    # Check for URLs in args
    if args:
        url_pattern = r'https?://[^\s]+'
        for arg in args:
            if re.match(url_pattern, arg):
                image_urls.append(arg)
                print(f"Found image URL: {arg}")  # Debug line

    if not image_urls:
        await message.reply("❌ Please attach an image or provide an image URL to analyze.")
        return

    await message.add_reaction("🔍")

    async with MediaHandler() as handler:
        for url in image_urls[:2]:  # Limit to 2 images
            try:
                result, error = await handler.process_media(url)
                if error:
                    await message.reply(f"❌ Error: {error}")
                    continue

                # Create AI prompt
                prompt = f"""Analyze this image:
- Format: {result.get('format', 'Unknown')}
- Size: {result.get('size', 'Unknown')}
- Text extracted: {result.get('text', 'No text found')}

Please describe what you can determine about this image."""

                # Get AI response (adjust this based on your AI client)
                ai_response = await get_ai_response(prompt, ai_client, message.author.id)

                response = f"🖼️ **Image Analysis:**\n{ai_response}"
                await message.reply(response[:2000])

            except Exception as e:
                await message.reply(f"❌ Error analyzing image: {str(e)}")

async def handle_file_analysis(message, args, ai_client):
    """Handle file analysis"""
    print("Handling file analysis...")  # Debug line

    file_urls = []

    # Check attachments
    for attachment in message.attachments:
        file_urls.append(attachment.url)
        print(f"Found file attachment: {attachment.url}")  # Debug line

    # Check for URLs in args
    if args:
        url_pattern = r'https?://[^\s]+'
        for arg in args:
            if re.match(url_pattern, arg):
                file_urls.append(arg)
                print(f"Found file URL: {arg}")  # Debug line

    if not file_urls:
        await message.reply("❌ Please attach a file or provide a file URL to analyze.")
        return

    await message.add_reaction("📄")

    async with MediaHandler() as handler:
        for url in file_urls[:2]:  # Limit to 2 files
            try:
                result, error = await handler.process_media(url)
                if error:
                    await message.reply(f"❌ Error: {error}")
                    continue

                # Create AI prompt
                prompt = f"""Analyze this file:
- Type: {result.get('type', 'Unknown')}
- Size: {result.get('size', 0)} bytes
- Content preview: {result.get('content', 'No readable content')[:1000]}

Please summarize what this file contains."""

                # Get AI response
                ai_response = await get_ai_response(prompt, ai_client, message.author.id)

                response = f"📄 **File Analysis:**\n{ai_response}"
                await message.reply(response[:2000])

            except Exception as e:
                await message.reply(f"❌ Error analyzing file: {str(e)}")

async def handle_link_analysis(message, args, ai_client):
    """Handle link analysis"""
    print("Handling link analysis...")  # Debug line

    if not args:
        await message.reply("❌ Please provide a URL. Usage: `!link <URL>`")
        return

    url = args[0]
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    await message.add_reaction("🔗")

    async with MediaHandler() as handler:
        try:
            result, error = await handler.scrape_webpage(url)
            if error:
                await message.reply(f"❌ Error: {error}")
                return

            # Create AI prompt
            prompt = f"""Analyze this webpage:
- Title: {result.get('title', 'No title')}
- URL: {result.get('url', url)}
- Content: {result.get('content', 'No content')[:2000]}

Please provide a summary of this webpage."""

            # Get AI response
            ai_response = await get_ai_response(prompt, ai_client, message.author.id)

            response = f"🔗 **Webpage Analysis:**\n**Title:** {result.get('title', 'No title')}\n{ai_response}"
            await message.reply(response[:2000])

        except Exception as e:
            await message.reply(f"❌ Error analyzing webpage: {str(e)}")

async def handle_search_analysis(message, args, ai_client):
    """Handle search analysis"""
    print("Handling search analysis...")  # Debug line

    if not args:
        await message.reply("❌ Please provide search terms. Usage: `!search <query>`")
        return

    query = ' '.join(args)
    await message.add_reaction("🔍")

    try:
        # Use your existing search function or create a simple one
        prompt = f"Please search for and provide information about: {query}"
        ai_response = await get_ai_response(prompt, ai_client, message.author.id)

        response = f"🔍 **Search Results for:** {query}\n\n{ai_response}"
        await message.reply(response[:2000])

    except Exception as e:
        await message.reply(f"❌ Error performing search: {str(e)}")

async def get_ai_response(prompt, ai_client, user_id):
    """Get AI response - adjust this based on your AI client structure"""
    try:
        # This is a generic version - you'll need to adjust based on your AI client
        if hasattr(ai_client, 'get_response'):
            return await ai_client.get_response(prompt, user_id)
        elif hasattr(ai_client, 'generate_response'):
            return await ai_client.generate_response(prompt, user_id)
        elif hasattr(ai_client, 'chat'):
            return await ai_client.chat(prompt, user_id)
        else:
            # Fallback - try to call it directly
            return await ai_client(prompt)
    except Exception as e:
        print(f"AI client error: {e}")
        return f"Error getting AI response: {str(e)}"

