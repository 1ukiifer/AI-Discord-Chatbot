"""
Memory module for Discord AI Bot
Handles conversation memory with persistence and cleanup
"""

import json
import os
import time
import threading
import asyncio
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)



class ConversationMemory:
    """Manages conversation memory for users with persistence and cleanup"""
    def add_channel_message(self, channel_id: str, user_id: str, username: str, content: str, message_id: str = None):
        """Add a message to channel history for context."""
        if not hasattr(self, 'channel_histories'):
            self.channel_histories = {}

        if channel_id not in self.channel_histories:
            self.channel_histories[channel_id] = []

        message_data = {
            "user_id": user_id,
            "username": username,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "message_id": message_id or f"live_{int(datetime.now().timestamp())}"
        }

        self.channel_histories[channel_id].append(message_data)

        # Keep only recent messages per channel (default 100)
        max_messages = getattr(self, 'max_channel_messages', 100)
        if len(self.channel_histories[channel_id]) > max_messages:
            self.channel_histories[channel_id] = self.channel_histories[channel_id][-max_messages:]
    def __init__(self, memory_file: str = 'conversation_memory.json', 
                 max_messages_per_user: int = 50, cleanup_days: int = 7):
        self.memory_file = memory_file
        self.max_messages_per_user = max_messages_per_user
        self.cleanup_days = cleanup_days
        self.conversations = defaultdict(lambda: deque(maxlen=max_messages_per_user))
        self.user_contexts = defaultdict(dict)
        self.lock = threading.Lock()
        
        # Load existing memory
        self.load_memory()
        
        # Schedule periodic cleanup
        self._schedule_cleanup()
    
    def add_message(self, user_id: str, message: str, response: str, channel_id: str = None):
        """Add a message-response pair to user's conversation history"""
        with self.lock:
            timestamp = datetime.now().isoformat()
            entry = {
                'timestamp': timestamp,
                'user_message': message,
                'bot_response': response,
                'channel_id': channel_id
            }
            
            self.conversations[user_id].append(entry)
            
            # Update user context
            if user_id not in self.user_contexts:
                self.user_contexts[user_id] = {
                    'first_interaction': timestamp,
                    'last_interaction': timestamp,
                    'total_messages': 0,
                    'preferred_topics': [],
                    'conversation_style': 'friendly'
                }
            
            self.user_contexts[user_id]['last_interaction'] = timestamp
            self.user_contexts[user_id]['total_messages'] += 1
            
            # Auto-save periodically
            if self.user_contexts[user_id]['total_messages'] % 10 == 0:
                self.save_memory()
    
    def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get recent conversation history for a user"""
        with self.lock:
            history = list(self.conversations[user_id])
            return history[-limit:] if history else []
    
    def get_context_for_prompt(self, user_id: str, current_message: str) -> str:
        """Generate context string to include in AI prompt"""
        history = self.get_conversation_history(user_id, limit=5)
        user_context = self.user_contexts.get(user_id, {})
        
        if not history:
            return ""
        
        context_parts = []
        
        # Add user context
        if user_context:
            total_msgs = user_context.get('total_messages', 0)
            context_parts.append(f"User context: This user has sent {total_msgs} messages total.")
            
            if user_context.get('preferred_topics'):
                topics = ', '.join(user_context['preferred_topics'])
                context_parts.append(f"User's interests: {topics}")
        
        # Add recent conversation history
        if history:
            context_parts.append("Recent conversation history:")
            for entry in history:
                # Truncate long messages
                user_msg = entry['user_message']
                if len(user_msg) > 200:
                    user_msg = user_msg[:200] + "..."
                
                bot_msg = entry['bot_response']
                if len(bot_msg) > 200:
                    bot_msg = bot_msg[:200] + "..."
                
                context_parts.append(f"User: {user_msg}")
                context_parts.append(f"Assistant: {bot_msg}")
        
        context_parts.append(f"Current message: {current_message}")
        context_parts.append("Please respond considering this conversation history and maintain consistency.")
        
        return "\n".join(context_parts)
    
    def update_user_interests(self, user_id: str, message: str):
        """Analyze message and update user's interests"""
        with self.lock:
            interest_keywords = [
                'programming', 'coding', 'python', 'javascript', 'gaming', 'music', 
                'movies', 'books', 'science', 'technology', 'art', 'cooking', 
                'travel', 'sports', 'fitness', 'photography', 'writing', 'ai',
                'machine learning', 'discord', 'bot', 'automation', 'web development'
            ]
            
            message_lower = message.lower()
            detected_interests = [kw for kw in interest_keywords if kw in message_lower]
            
            if detected_interests and user_id in self.user_contexts:
                current_interests = self.user_contexts[user_id].get('preferred_topics', [])
                
                for interest in detected_interests:
                    if interest not in current_interests:
                        current_interests.append(interest)
                
                # Keep only last 10 interests
                self.user_contexts[user_id]['preferred_topics'] = current_interests[-10:]
    
    def get_user_summary(self, user_id: str) -> Dict:
        """Get a summary of user's conversation patterns"""
        with self.lock:
            history = list(self.conversations[user_id])
            context = self.user_contexts.get(user_id, {})
            
            if not history:
                return {"message": "No conversation history found."}
            
            total_messages = len(history)
            avg_length = sum(len(entry['user_message']) for entry in history) / total_messages
            
            recent_activity = []
            for entry in history[-5:]:
                preview = entry['user_message']
                if len(preview) > 100:
                    preview = preview[:100] + "..."
                
                recent_activity.append({
                    'timestamp': entry['timestamp'],
                    'preview': preview
                })
            
            return {
                'total_messages': total_messages,
                'avg_message_length': round(avg_length, 1),
                'first_interaction': context.get('first_interaction'),
                'last_interaction': context.get('last_interaction'),
                'interests': context.get('preferred_topics', []),
                'recent_activity': recent_activity
            }
    
    def clear_user_memory(self, user_id: str):
        """Clear all memory for a specific user"""
        with self.lock:
            if user_id in self.conversations:
                del self.conversations[user_id]
            if user_id in self.user_contexts:
                del self.user_contexts[user_id]
            self.save_memory()
    
    def save_memory(self):
        """Save conversation memory to file"""
        try:
            with self.lock:
                data = {
                    'conversations': {
                        user_id: list(messages) 
                        for user_id, messages in self.conversations.items()
                    },
                    'user_contexts': dict(self.user_contexts),
                    'last_saved': datetime.now().isoformat()
                }
                
                with open(self.memory_file, 'w') as f:
                    json.dump(data, f, indent=2)
                    
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
    
    def load_memory(self):
        """Load conversation memory from file"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                
                # Restore conversations
                for user_id, messages in data.get('conversations', {}).items():
                    self.conversations[user_id] = deque(messages, maxlen=self.max_messages_per_user)
                
                # Restore user contexts
                for user_id, context in data.get('user_contexts', {}).items():
                    self.user_contexts[user_id] = context
                
                logger.info(f"Loaded memory for {len(self.conversations)} users")
                
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")
    
    def cleanup_old_conversations(self):
        """Remove conversations older than cleanup_days"""
        cutoff_date = datetime.now() - timedelta(days=self.cleanup_days)
        
        with self.lock:
            users_to_remove = []
            
            for user_id, context in self.user_contexts.items():
                try:
                    last_interaction = datetime.fromisoformat(context['last_interaction'])
                    if last_interaction < cutoff_date:
                        users_to_remove.append(user_id)
                except (KeyError, ValueError):
                    users_to_remove.append(user_id)
            
            for user_id in users_to_remove:
                if user_id in self.conversations:
                    del self.conversations[user_id]
                if user_id in self.user_contexts:
                    del self.user_contexts[user_id]
            
            if users_to_remove:
                logger.info(f"Cleaned up memory for {len(users_to_remove)} inactive users")
                self.save_memory()
    
    def _schedule_cleanup(self):
        """Schedule periodic cleanup"""
        def cleanup_task():
            while True:
                time.sleep(24 * 3600)  # Run daily
                self.cleanup_old_conversations()
        
        cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
        cleanup_thread.start()
    
    def get_memory_stats(self) -> Dict:
        """Get overall memory statistics"""
        with self.lock:
            total_users = len(self.conversations)
            total_messages = sum(len(conv) for conv in self.conversations.values())
            
            file_size = 0
            if os.path.exists(self.memory_file):
                file_size = os.path.getsize(self.memory_file)
            
            return {
                'total_users': total_users,
                'total_messages': total_messages,
                'memory_file_size_kb': round(file_size / 1024, 2),
                'max_messages_per_user': self.max_messages_per_user,
                'cleanup_days': self.cleanup_days
            }

class AIServiceWithMemory:
    """AI service wrapper that adds conversation memory"""
    
    def __init__(self, memory_instance, ai_service):
        self.memory = memory_instance
        self.ai_service = ai_service
    
    async def generate_response(self, prompt: str, user_id: Optional[str] = None) -> str:
        """Generate AI response with conversation memory"""
        try:
            # Get conversation context
            enhanced_prompt = prompt
            if user_id:
                context = self.memory.get_context_for_prompt(str(user_id), prompt)
                if context:
                    enhanced_prompt = f"{context}\n\nBased on our conversation history, please respond to: {prompt}"
                
                # Update user interests
                self.memory.update_user_interests(str(user_id), prompt)
            
            # Generate response using the AI service
            response = await self.ai_service.generate_response(enhanced_prompt, user_id)
            
            # Store conversation in memory
            if user_id:
                self.memory.add_message(str(user_id), prompt, response)
            
            return response
                
        except Exception as e:
            logger.error(f"AI generation with memory error: {e}")
            return "Sorry, I encountered an error generating a response. Please try again."

