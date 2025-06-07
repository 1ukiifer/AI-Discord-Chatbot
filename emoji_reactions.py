"""
Emoji Reactions module for Discord AI Bot
Provides context-aware emoji reactions to bot responses
"""

import logging
import random
from typing import List, Optional
import discord
from config import Config

logger = logging.getLogger(__name__)

class EmojiReactionService:
    """Service for adding context-aware emoji reactions to bot responses"""
    
    def __init__(self):
        self.reaction_patterns = {
            # Positive responses
            'positive': ['👍', '✅', '😊', '🎉', '💯', '🌟'],
            'helpful': ['🤝', '💡', '📚', '🔧', '🎯', '✨'],
            'creative': ['🎨', '🌈', '💭', '🚀', '⭐', '🎪'],
            
            # Topic-specific
            'programming': ['💻', '⌨️', '🖥️', '🔧', '⚙️', '🐛'],
            'music': ['🎵', '🎶', '🎤', '🎸', '🥁', '🎹'],
            'gaming': ['🎮', '🕹️', '🎯', '🏆', '⚔️', '🎲'],
            'science': ['🔬', '⚗️', '🧬', '🌌', '🔭', '⚡'],
            'food': ['🍕', '🍔', '🍰', '☕', '🥗', '🍳'],
            'travel': ['✈️', '🗺️', '🌍', '🚗', '🏔️', '🏖️'],
            'books': ['📖', '📚', '✍️', '📝', '📰', '🔖'],
            
            # Emotional context
            'thinking': ['🤔', '💭', '🧠', '⚡', '💡'],
            'confused': ['❓', '🤷', '😅', '🔍'],
            'excited': ['🎉', '🚀', '⭐', '🌟', '💫'],
            'search': ['🔍', '🔎', '📊', '📈', '🗂️'],
            
            # General responses
            'greeting': ['👋', '😊', '🌅', '☀️'],
            'farewell': ['👋', '🌙', '✨', '💫'],
            'question': ['❓', '🤔', '💭', '🔍'],
            'information': ['📊', '📋', '📈', '💼', '🗂️']
        }
    
    async def add_context_reactions(self, message: discord.Message, response_content: str) -> None:
        """Add context-aware reactions to the bot's response message"""
        if not Config.ENABLE_EMOJI_REACTIONS:
            return
        
        try:
            # Analyze response content for context
            reactions = self._analyze_content_for_reactions(response_content)
            
            # Add reactions (limit to 3 to avoid spam)
            for reaction in reactions[:3]:
                try:
                    await message.add_reaction(reaction)
                except discord.HTTPException:
                    # Skip if reaction fails (maybe emoji not available)
                    continue
                    
        except Exception as e:
            logger.error(f"Error adding emoji reactions: {e}")
    
    def _analyze_content_for_reactions(self, content: str) -> List[str]:
        """Analyze content and return appropriate emoji reactions"""
        content_lower = content.lower()
        reactions = []
        
        # Check for topic-specific keywords
        topic_keywords = {
            'programming': ['code', 'python', 'javascript', 'programming', 'software', 'bug', 'function', 'variable'],
            'music': ['song', 'music', 'album', 'artist', 'melody', 'rhythm', 'instrument'],
            'gaming': ['game', 'play', 'player', 'level', 'score', 'console', 'pc gaming'],
            'science': ['research', 'experiment', 'theory', 'data', 'analysis', 'discovery'],
            'food': ['recipe', 'cooking', 'food', 'eat', 'meal', 'ingredient', 'restaurant'],
            'travel': ['trip', 'travel', 'vacation', 'destination', 'flight', 'hotel'],
            'books': ['book', 'read', 'author', 'story', 'novel', 'chapter', 'literature']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                reactions.extend(random.sample(self.reaction_patterns[topic], min(2, len(self.reaction_patterns[topic]))))
                break
        
        # Check for emotional context
        if any(word in content_lower for word in ['think', 'consider', 'analyze', 'evaluate']):
            reactions.extend(random.sample(self.reaction_patterns['thinking'], 1))
        
        if any(word in content_lower for word in ['search', 'find', 'look', 'information']):
            reactions.extend(random.sample(self.reaction_patterns['search'], 1))
        
        if any(word in content_lower for word in ['excited', 'amazing', 'awesome', 'fantastic']):
            reactions.extend(random.sample(self.reaction_patterns['excited'], 1))
        
        if any(word in content_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            reactions.extend(random.sample(self.reaction_patterns['greeting'], 1))
        
        if any(word in content_lower for word in ['bye', 'goodbye', 'farewell', 'see you']):
            reactions.extend(random.sample(self.reaction_patterns['farewell'], 1))
        
        # Check for questions
        if '?' in content or any(word in content_lower for word in ['what', 'how', 'why', 'when', 'where']):
            reactions.extend(random.sample(self.reaction_patterns['question'], 1))
        
        # Check for positive sentiment
        if any(word in content_lower for word in ['great', 'good', 'excellent', 'perfect', 'wonderful']):
            reactions.extend(random.sample(self.reaction_patterns['positive'], 1))
        
        # Check for helpful content
        if any(word in content_lower for word in ['help', 'assist', 'guide', 'tutorial', 'explain']):
            reactions.extend(random.sample(self.reaction_patterns['helpful'], 1))
        
        # Default reactions if none found
        if not reactions:
            reactions = random.sample(self.reaction_patterns['information'], 1)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_reactions = []
        for reaction in reactions:
            if reaction not in seen:
                unique_reactions.append(reaction)
                seen.add(reaction)
        
        return unique_reactions
    
    def get_personality_reactions(self, personality_mode: str) -> List[str]:
        """Get reactions based on personality mode"""
        personality_reactions = {
            'professional': ['✅', '📊', '💼', '🎯'],
            'casual': ['😊', '👍', '🎉', '😄'],
            'technical': ['⚙️', '🔧', '💻', '🧠'],
            'adaptive': ['✨', '🌟', '💫', '🎭']
        }
        
        return personality_reactions.get(personality_mode, personality_reactions['adaptive'])