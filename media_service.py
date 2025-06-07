"""
Media Service for Discord AI Bot
Handles image, video, and document processing with AI analysis
"""
import os
import aiohttp
import asyncio
import logging
from typing import Optional, Dict, List, Any, Tuple
from PIL import Image, ImageEnhance
import io
import base64
import mimetypes
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

class MediaService:
    """Service for processing and analyzing media content from Discord messages."""

    def __init__(self, max_file_size_mb: int = 25, cache_dir: str = "media_cache"):
        """
        Initialize the media service.

        Args:
            max_file_size_mb: Maximum file size to process in MB
            cache_dir: Directory to cache processed media
        """
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.cache_dir = cache_dir
        self.supported_image_types = {
            'image/jpeg', 'image/jpg', 'image/png', 'image/gif', 
            'image/webp', 'image/bmp', 'image/tiff'
        }
        self.supported_video_types = {
            'video/mp4', 'video/avi', 'video/mov', 'video/wmv',
            'video/flv', 'video/webm', 'video/mkv'
        }
        self.supported_document_types = {
            'application/pdf', 'text/plain', 'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/vnd.ms-excel',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        }

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

        logger.info(f"Media service initialized with max file size: {max_file_size_mb}MB")

    def _get_file_hash(self, content: bytes) -> str:
        """Generate SHA-256 hash of file content for caching."""
        return hashlib.sha256(content).hexdigest()

    def _get_cache_path(self, file_hash: str, extension: str = "") -> str:
        """Get cache file path for a given hash."""
        return os.path.join(self.cache_dir, f"{file_hash}{extension}")

    async def download_media(self, url: str) -> Optional[Tuple[bytes, str]]:
        """
        Download media from URL.

        Args:
            url: Media URL to download

        Returns:
            Tuple of (content_bytes, content_type) or None if failed
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to download media: HTTP {response.status}")
                        return None

                    content_length = response.headers.get('content-length')
                    if content_length and int(content_length) > self.max_file_size_bytes:
                        logger.warning(f"File too large: {content_length} bytes")
                        return None

                    content = await response.read()
                    if len(content) > self.max_file_size_bytes:
                        logger.warning(f"Downloaded file too large: {len(content)} bytes")
                        return None

                    content_type = response.headers.get('content-type', '').lower()
                    return content, content_type

        except Exception as e:
            logger.error(f"Error downloading media from {url}: {e}")
            return None

    def is_supported_media(self, content_type: str) -> bool:
        """Check if the content type is supported for processing."""
        content_type = content_type.lower().split(';')[0]  # Remove charset info
        return (content_type in self.supported_image_types or 
                content_type in self.supported_video_types or
                content_type in self.supported_document_types)

    def get_media_type(self, content_type: str) -> str:
        """Get the general media type category."""
        content_type = content_type.lower().split(';')[0]
        if content_type in self.supported_image_types:
            return "image"
        elif content_type in self.supported_video_types:
            return "video"
        elif content_type in self.supported_document_types:
            return "document"
        else:
            return "unknown"

    async def process_image(self, content: bytes, content_type: str) -> Optional[Dict[str, Any]]:
        """
        Process an image for AI analysis.

        Args:
            content: Image bytes
            content_type: MIME type of the image

        Returns:
            Dictionary containing processed image data and metadata
        """
        try:
            file_hash = self._get_file_hash(content)
            cache_path = self._get_cache_path(file_hash, ".json")

            # Check cache first
            if os.path.exists(cache_path):
                logger.debug(f"Loading cached image analysis: {file_hash}")
                # In a real implementation, you'd load cached analysis here
                pass

            # Process image with PIL
            image = Image.open(io.BytesIO(content))

            # Basic image info
            width, height = image.size
            format_name = image.format
            mode = image.mode

            # Convert to RGB if necessary for AI processing
            if image.mode in ('RGBA', 'LA', 'P'):
                # Create a white background for transparent images
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                background.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')

            # Resize if too large (for AI processing efficiency)
            max_dimension = 1024
            if max(width, height) > max_dimension:
                ratio = max_dimension / max(width, height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")

            # Convert to base64 for AI services
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=85)
            base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Extract basic metadata
            metadata = {
                'original_size': (width, height),
                'current_size': image.size,
                'format': format_name,
                'mode': mode,
                'file_size': len(content),
                'processed_at': datetime.now().isoformat(),
                'file_hash': file_hash
            }

            # Try to extract EXIF data if available
            try:
                if hasattr(image, '_getexif') and image._getexif():
                    exif_data = image._getexif()
                    metadata['exif'] = {k: str(v) for k, v in exif_data.items() if isinstance(v, (str, int, float))}
            except Exception as e:
                logger.debug(f"Could not extract EXIF data: {e}")

            result = {
                'type': 'image',
                'base64_data': base64_image,
                'metadata': metadata,
                'content_type': 'image/jpeg',  # Always JPEG after processing
                'analysis_ready': True
            }

            logger.info(f"Processed image: {width}x{height} {format_name}")
            return result

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return None

    async def process_video(self, content: bytes, content_type: str) -> Optional[Dict[str, Any]]:
        """
        Process a video file (extract metadata, thumbnails, etc.).

        Args:
            content: Video bytes
            content_type: MIME type of the video

        Returns:
            Dictionary containing video metadata
        """
        try:
            file_hash = self._get_file_hash(content)

            # For now, just return basic metadata
            # In a full implementation, you'd use ffmpeg or similar to extract frames
            metadata = {
                'file_size': len(content),
                'content_type': content_type,
                'processed_at': datetime.now().isoformat(),
                'file_hash': file_hash
            }

            result = {
                'type': 'video',
                'metadata': metadata,
                'content_type': content_type,
                'analysis_ready': False,  # Video analysis not implemented yet
                'description': f"Video file ({content_type}, {len(content)} bytes)"
            }

            logger.info(f"Processed video metadata: {content_type}, {len(content)} bytes")
            return result

        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return None

    async def process_document(self, content: bytes, content_type: str) -> Optional[Dict[str, Any]]:
        """
        Process a document file (extract text, metadata, etc.).

        Args:
            content: Document bytes
            content_type: MIME type of the document

        Returns:
            Dictionary containing document data and metadata
        """
        try:
            file_hash = self._get_file_hash(content)
            text_content = ""

            # Extract text based on content type
            if content_type == 'text/plain':
                try:
                    text_content = content.decode('utf-8', errors='ignore')
                except:
                    text_content = content.decode('latin-1', errors='ignore')

            elif content_type == 'application/pdf':
                # For PDF processing, you'd typically use PyPDF2 or pdfplumber
                # This is a placeholder implementation
                text_content = f"[PDF Document - {len(content)} bytes - Text extraction not implemented]"

            elif 'word' in content_type or 'officedocument' in content_type:
                # For Word documents, you'd use python-docx or similar
                text_content = f"[Word Document - {len(content)} bytes - Text extraction not implemented]"

            elif 'excel' in content_type or 'spreadsheet' in content_type:
                # For Excel files, you'd use openpyxl or pandas
                text_content = f"[Excel Document - {len(content)} bytes - Data extraction not implemented]"

            metadata = {
                'file_size': len(content),
                'content_type': content_type,
                'text_length': len(text_content),
                'processed_at': datetime.now().isoformat(),
                'file_hash': file_hash
            }

            result = {
                'type': 'document',
                'text_content': text_content[:5000],  # Limit text length
                'metadata': metadata,
                'content_type': content_type,
                'analysis_ready': True
            }

            logger.info(f"Processed document: {content_type}, {len(text_content)} chars extracted")
            return result

        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return None

    async def analyze_media(self, media_data: Dict[str, Any]) -> Optional[str]:
        """
        Generate a description of the media for AI context.

        Args:
            media_data: Processed media data from process_* methods

        Returns:
            Human-readable description of the media
        """
        try:
            media_type = media_data.get('type', 'unknown')
            metadata = media_data.get('metadata', {})

            if media_type == 'image':
                size = metadata.get('current_size', metadata.get('original_size', (0, 0)))
                format_name = metadata.get('format', 'unknown')
                description = f"Image: {size[0]}x{size[1]} pixels, {format_name} format"

                # Add any EXIF information if available
                if 'exif' in metadata:
                    description += " (contains EXIF metadata)"

                return description

            elif media_type == 'video':
                file_size = metadata.get('file_size', 0)
                content_type = metadata.get('content_type', 'unknown')
                return f"Video file: {content_type}, {file_size // 1024} KB"

            elif media_type == 'document':
                content_type = metadata.get('content_type', 'unknown')
                text_length = metadata.get('text_length', 0)
                if text_length > 0:
                    return f"Document: {content_type}, {text_length} characters of text"
                else:
                    return f"Document: {content_type} (binary content)"

            return f"Media file: {media_type}"

        except Exception as e:
            logger.error(f"Error analyzing media: {e}")
            return "Media file (analysis failed)"

    async def process_attachment(self, attachment_url: str, filename: str = "") -> Optional[Dict[str, Any]]:
        """
        Process a Discord attachment by URL.

        Args:
            attachment_url: URL of the Discord attachment
            filename: Original filename (optional, for type detection)

        Returns:
            Processed media data or None if failed
        """
        try:
            # Download the media
            download_result = await self.download_media(attachment_url)
            if not download_result:
                return None

            content, content_type = download_result

            # If content type is not detected, try to infer from filename
            if not content_type or content_type == 'application/octet-stream':
                if filename:
                    guessed_type, _ = mimetypes.guess_type(filename)
                    if guessed_type:
                        content_type = guessed_type

            # Check if we support this media type
            if not self.is_supported_media(content_type):
                logger.warning(f"Unsupported media type: {content_type}")
                return {
                    'type': 'unsupported',
                    'content_type': content_type,
                    'filename': filename,
                    'file_size': len(content),
                    'analysis_ready': False
                }

            # Process based on media type
            media_type = self.get_media_type(content_type)

            if media_type == 'image':
                return await self.process_image(content, content_type)
            elif media_type == 'video':
                return await self.process_video(content, content_type)
            elif media_type == 'document':
                return await self.process_document(content, content_type)

            return None

        except Exception as e:
            logger.error(f"Error processing attachment {attachment_url}: {e}")
            return None

    async def cleanup_cache(self, max_age_days: int = 7):
        """
        Clean up old cached files.

        Args:
            max_age_days: Maximum age of cached files in days
        """
        try:
            import time
            current_time = time.time()
            max_age_seconds = max_age_days * 24 * 60 * 60

            removed_count = 0
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > max_age_seconds:
                        os.remove(file_path)
                        removed_count += 1

            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old cached media files")

        except Exception as e:
            logger.error(f"Error cleaning up media cache: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the media cache."""
        try:
            cache_files = os.listdir(self.cache_dir)
            total_size = sum(
                os.path.getsize(os.path.join(self.cache_dir, f)) 
                for f in cache_files 
                if os.path.isfile(os.path.join(self.cache_dir, f))
            )

            return {
                'cache_dir': self.cache_dir,
                'file_count': len(cache_files),
                'total_size_kb': total_size // 1024,
                'max_file_size_mb': self.max_file_size_bytes // (1024 * 1024)
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {'error': str(e)}