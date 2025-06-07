"""
AI Service module for Discord AI Bot
Handles multiple AI providers (OpenAI, Anthropic, Gemini)
"""

import logging
import asyncio
from typing import Optional
import google.generativeai as genai
import openai
import anthropic
from config import Config

logger = logging.getLogger(__name__)

class AIService:
    """AI service supporting multiple providers"""

    def __init__(self):
        self.provider = Config.AI_PROVIDER
        self.available_providers = ['openai', 'anthropic', 'gemini']
        self._setup_clients()

    def switch_provider(self, new_provider: str) -> bool:
        """Switch AI provider if credentials are available"""
        if new_provider not in self.available_providers:
            return False

        # Check if credentials are available
        if new_provider == 'openai' and not Config.OPENAI_API_KEY:
            return False
        elif new_provider == 'anthropic' and not Config.ANTHROPIC_API_KEY:
            return False
        elif new_provider == 'gemini' and not Config.GEMINI_API_KEY:
            return False

        self.provider = new_provider
        self._setup_clients()
        return True

    def get_available_providers(self) -> list:
        """Get list of providers with available credentials"""
        available = []
        if Config.OPENAI_API_KEY:
            available.append('openai')
        if Config.ANTHROPIC_API_KEY:
            available.append('anthropic')
        if Config.GEMINI_API_KEY:
            available.append('gemini')
        return available

    def _setup_clients(self):
        """Initialize AI service clients based on configuration"""
        try:
            if self.provider == 'openai':
                # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
                self.openai_client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
                logger.info("OpenAI client initialized")

            elif self.provider == 'anthropic':
                # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
                self.anthropic_client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
                logger.info("Anthropic client initialized")

            elif self.provider == 'gemini':
                genai.configure(api_key=Config.GEMINI_API_KEY)
                self.gemini_model = genai.GenerativeModel(Config.GEMINI_MODEL)
                logger.info("Gemini client initialized")

        except Exception as e:
            logger.error(f"Failed to initialize AI client for {self.provider}: {e}")
            raise

    async def generate_response(self, prompt: str, user_id: Optional[str] = None) -> str:
        """Generate AI response using the configured provider"""
        try:
            # Get personality-enhanced system prompt
            # NOTE: Search context is now handled in main.py and passed via prompt
            system_prompt = self._get_personality_prompt()

            logger.info(f"Generating {self.provider} response for user {user_id}")
            logger.info(f"Prompt length: {len(prompt)} characters")

            if self.provider == 'openai':
                return await self._generate_openai_response(prompt, system_prompt)
            elif self.provider == 'anthropic':
                return await self._generate_anthropic_response(prompt, system_prompt)
            elif self.provider == 'gemini':
                return await self._generate_gemini_response(prompt, system_prompt)
            else:
                raise ValueError(f"Unsupported AI provider: {self.provider}")

        except Exception as e:
            logger.error(f"AI generation error with {self.provider}: {e}", exc_info=True)
            return "Sorry, I encountered an error generating a response. Please try again."

    def _get_personality_prompt(self) -> str:
        """Get personality-enhanced system prompt"""
        base_prompt = Config.SYSTEM_PROMPT

        personality_additions = {
            'professional': " Maintain a professional, formal tone in all responses.",
            'casual': " Use a relaxed, friendly tone with occasional internet slang when appropriate.",
            'technical': " Focus on technical accuracy and provide detailed explanations when relevant.",
            'adaptive': " Adapt your tone to match the user's communication style and context."
        }

        personality_mode = Config.PERSONALITY_MODE
        if personality_mode in personality_additions:
            return base_prompt + personality_additions[personality_mode]

        return base_prompt

    async def _generate_openai_response(self, prompt: str, system_prompt: str) -> str:
        """Generate response using OpenAI"""
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=Config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=Config.MAX_TOKENS,
                temperature=Config.TEMPERATURE
            )

            result = response.choices[0].message.content.strip()
            logger.info(f"OpenAI response generated: {len(result)} characters")
            return result

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    async def _generate_anthropic_response(self, prompt: str, system_prompt: str) -> str:
        """Generate response using Anthropic Claude"""
        try:
            response = await asyncio.to_thread(
                self.anthropic_client.messages.create,
                model=Config.ANTHROPIC_MODEL,
                max_tokens=Config.MAX_TOKENS,
                temperature=Config.TEMPERATURE,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )

            result = response.content[0].text.strip()
            logger.info(f"Anthropic response generated: {len(result)} characters")
            return result

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    async def _generate_gemini_response(self, prompt: str, system_prompt: str) -> str:
        """Generate response using Google Gemini"""
        try:
            # Combine system prompt with user prompt for Gemini
            full_prompt = f"{system_prompt}\n\nUser: {prompt}"

            response = await asyncio.to_thread(
                self.gemini_model.generate_content,
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=Config.MAX_TOKENS,
                    temperature=Config.TEMPERATURE
                )
            )

            result = response.text.strip()
            logger.info(f"Gemini response generated: {len(result)} characters")
            return result

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    def get_provider_info(self) -> dict:
        """Get information about the current AI provider"""
        model_map = {
            'openai': Config.OPENAI_MODEL,
            'anthropic': Config.ANTHROPIC_MODEL,
            'gemini': Config.GEMINI_MODEL
        }

        return {
            'provider': self.provider,
            'model': model_map.get(self.provider, 'unknown'),
            'max_tokens': Config.MAX_TOKENS,
            'temperature': Config.TEMPERATURE
        }


# Add these methods to your existing AIService class in ai_service.py

async def generate_response_with_media(self, text: str, media_data: list[dict[str, any]] = None, user_id: str = None) -> str:
    """
    Generate AI response with media context.

    Args:
        text: User's text message
        media_data: List of processed media data from MediaService
        user_id: User identifier for personalization

    Returns:
        AI-generated response considering media context
    """
    try:
        # Build media context
        media_context = ""
        if media_data:
            media_context = "\n\nMedia attachments in this message:\n"
            for i, media in enumerate(media_data, 1):
                if media.get('analysis_ready', False):
                    media_type = media.get('type', 'unknown')

                    if media_type == 'image':
                        media_context += f"{i}. {await self._describe_image_media(media)}\n"
                    elif media_type == 'document':
                        media_context += f"{i}. {await self._describe_document_media(media)}\n"
                    elif media_type == 'video':
                        media_context += f"{i}. {await self._describe_video_media(media)}\n"
                    else:
                        media_context += f"{i}. {media.get('description', 'Unknown media type')}\n"
                else:
                    media_context += f"{i}. {media.get('description', 'Media file (not analyzed)')}\n"

        # Combine text and media context
        full_prompt = text + media_context if media_context else text

        # Generate response based on current provider
        if self.current_provider == 'openai':
            return await self._generate_openai_response_with_media(full_prompt, media_data, user_id)
        elif self.current_provider == 'anthropic':
            return await self._generate_anthropic_response_with_media(full_prompt, media_data, user_id)
        elif self.current_provider == 'gemini':
            return await self._generate_gemini_response_with_media(full_prompt, media_data, user_id)
        else:
            # Fallback to text-only response
            return await self.generate_response(full_prompt, user_id)

    except Exception as e:
        logger.error(f"Error generating response with media: {e}")
        return "I encountered an error while processing your message and attachments. Please try again."

async def _describe_image_media(self, media: dict[str, any]) -> str:
    """Generate description for image media."""
    metadata = media.get('metadata', {})
    size = metadata.get('current_size', (0, 0))
    format_name = metadata.get('format', 'unknown')

    description = f"Image ({format_name}, {size[0]}x{size[1]} pixels)"

    # Add any relevant metadata
    if metadata.get('exif'):
        description += " with EXIF data"

    return description

async def _describe_document_media(self, media: dict[str, any]) -> str:
    """Generate description for document media."""
    metadata = media.get('metadata', {})
    content_type = metadata.get('content_type', 'unknown')
    text_length = metadata.get('text_length', 0)

    if text_length > 0:
        preview = media.get('text_content', '')[:200]
        if len(preview) < len(media.get('text_content', '')):
            preview += "..."
        return f"Document ({content_type}, {text_length} chars): {preview}"
    else:
        return f"Document ({content_type})"

async def _describe_video_media(self, media: dict[str, any]) -> str:
    """Generate description for video media."""
    metadata = media.get('metadata', {})
    content_type = metadata.get('content_type', 'unknown')
    file_size = metadata.get('file_size', 0)

    return f"Video ({content_type}, {file_size // 1024} KB)"

async def _generate_openai_response_with_media(self, prompt: str, media_data: list[dict[str, any]], user_id: str = None) -> str:
    """Generate OpenAI response with media support."""
    try:
        messages = [{"role": "system", "content": self._get_system_prompt()}]

        # Check if any media contains images for vision models
        has_images = any(media.get('type') == 'image' and media.get('analysis_ready') for media in media_data or [])

        if has_images and self.config.get('OPENAI_MODEL', '').startswith('gpt-4'):
            # Use vision model for images
            content = [{"type": "text", "text": prompt}]

            # Add images to the message
            for media in media_data or []:
                if media.get('type') == 'image' and media.get('base64_data'):
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{media['base64_data']}",
                            "detail": "auto"
                        }
                    })

            messages.append({"role": "user", "content": content})
        else:
            # Text-only response
            messages.append({"role": "user", "content": prompt})

        response = await self.openai_client.chat.completions.create(
            model=self.config.get('OPENAI_MODEL', 'gpt-3.5-turbo'),
            messages=messages,
            max_tokens=self.config.get('MAX_TOKENS', 1000),
            temperature=self.config.get('TEMPERATURE', 0.7)
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"OpenAI API error with media: {e}")
        raise

async def _generate_anthropic_response_with_media(self, prompt: str, media_data: list[dict[str, any]], user_id: str = None) -> str:
    """Generate Anthropic Claude response with media support."""
    try:
        # Claude supports image analysis
        content = [{"type": "text", "text": prompt}]

        # Add images if present
        for media in media_data or []:
            if media.get('type') == 'image' and media.get('base64_data'):
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": media['base64_data']
                    }
                })

        response = await self.anthropic_client.messages.create(
            model=self.config.get('ANTHROPIC_MODEL', 'claude-3-haiku-20240307'),
            system=self._get_system_prompt(),
            messages=[{"role": "user", "content": content}],
            max_tokens=self.config.get('MAX_TOKENS', 1000),
            temperature=self.config.get('TEMPERATURE', 0.7)
        )

        return response.content[0].text.strip()

    except Exception as e:
        logger.error(f"Anthropic API error with media: {e}")
        raise

async def _generate_gemini_response_with_media(self, prompt: str, media_data: list[dict[str, any]], user_id: str = None) -> str:
    """Generate Gemini response with media support."""
    try:
        import google.generativeai as genai

        model = genai.GenerativeModel(self.config.get('GEMINI_MODEL', 'gemini-pro'))

        # Check if we have images for Gemini Vision
        has_images = any(media.get('type') == 'image' and media.get('analysis_ready') for media in media_data or [])

        if has_images:
            # Use gemini-pro-vision for images
            model = genai.GenerativeModel('gemini-pro-vision')

            # Prepare content with images
            content_parts = [prompt]

            for media in media_data or []:
                if media.get('type') == 'image' and media.get('base64_data'):
                    # Convert base64 to PIL Image for Gemini
                    import base64
                    from PIL import Image
                    import io

                    image_data = base64.b64decode(media['base64_data'])
                    image = Image.open(io.BytesIO(image_data))
                    content_parts.append(image)

            response = await model.generate_content_async(content_parts)
        else:
            # Text-only response
            response = await model.generate_content_async(prompt)

        return response.text.strip()

    except Exception as e:
        logger.error(f"Gemini API error with media: {e}")
        raise

def supports_vision(self) -> bool:
    """Check if current AI provider supports vision/image analysis."""
    if self.current_provider == 'openai':
        return self.config.get('OPENAI_MODEL', '').startswith('gpt-4')
    elif self.current_provider == 'anthropic':
        return True  # Claude supports vision
    elif self.current_provider == 'gemini':
        return True  # Gemini has vision models
    return False

def get_media_capabilities(self) -> dict[str, any]:
    """Get information about media processing capabilities."""
    return {
        'provider': self.current_provider,
        'supports_vision': self.supports_vision(),
        'supported_image_types': ['jpeg', 'png', 'gif', 'webp'],
        'max_images_per_message': 10,
        'supports_documents': True,
        'supports_video_metadata': True
    }