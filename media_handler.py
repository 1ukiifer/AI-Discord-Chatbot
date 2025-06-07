import os
import aiohttp
import aiofiles
import asyncio
from PIL import Image
import pytesseract
import tempfile
import logging
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import re
import mimetypes

logger = logging.getLogger(__name__)

class MediaHandler:
    def __init__(self):
        self.session = None
        self.temp_dir = tempfile.mkdtemp()

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def download_file(self, url, max_size=50*1024*1024):  # 50MB limit
        """Download file from URL with size limit"""
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return None, f"Failed to download: HTTP {response.status}"

                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > max_size:
                    return None, "File too large (max 50MB)"

                content = await response.read()
                if len(content) > max_size:
                    return None, "File too large (max 50MB)"

                # Create temp file
                temp_path = os.path.join(self.temp_dir, f"temp_{asyncio.get_event_loop().time()}")
                async with aiofiles.open(temp_path, 'wb') as f:
                    await f.write(content)

                return temp_path, None
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            return None, f"Download error: {str(e)}"

    async def analyze_image(self, file_path):
        """Extract text from image using OCR"""
        try:
            # Open and process image
            with Image.open(file_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Extract text using OCR
                text = pytesseract.image_to_string(img)

                # Get basic image info
                info = {
                    'format': img.format,
                    'size': img.size,
                    'mode': img.mode,
                    'text': text.strip() if text.strip() else "No text found in image"
                }

                return info, None
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return None, f"Image analysis error: {str(e)}"

    async def analyze_document(self, file_path):
        """Analyze document files"""
        try:
            # Get file type using mimetypes
            file_type, _ = mimetypes.guess_type(file_path)
            if not file_type:
                # Try to determine by extension
                ext = os.path.splitext(file_path)[1].lower()
                if ext in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml']:
                    file_type = 'text/plain'
                elif ext in ['.pdf']:
                    file_type = 'application/pdf'
                elif ext in ['.doc', '.docx']:
                    file_type = 'application/msword'
                else:
                    file_type = 'application/octet-stream'

            file_size = os.path.getsize(file_path)

            content = ""

            # Read text files
            if file_type and (file_type.startswith('text/') or file_type in ['application/json', 'application/xml']):
                try:
                    async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = await f.read()
                        # Limit content length
                        if len(content) > 5000:
                            content = content[:5000] + "... (truncated)"
                except:
                    # If UTF-8 fails, try with latin-1
                    try:
                        async with aiofiles.open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
                            content = await f.read()
                            if len(content) > 5000:
                                content = content[:5000] + "... (truncated)"
                    except:
                        content = "Could not read file content"

            info = {
                'type': file_type,
                'size': file_size,
                'content': content if content else "Binary file - content not readable as text"
            }

            return info, None
        except Exception as e:
            logger.error(f"Error analyzing document: {e}")
            return None, f"Document analysis error: {str(e)}"

    async def scrape_webpage(self, url):
        """Scrape and analyze webpage content"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            async with self.session.get(url, headers=headers) as response:
                if response.status != 200:
                    return None, f"Failed to fetch webpage: HTTP {response.status}"

                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')

                # Extract basic info
                title = soup.find('title')
                title = title.get_text().strip() if title else "No title found"

                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()

                # Get text content
                text = soup.get_text()

                # Clean up text
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)

                # Limit text length
                if len(text) > 3000:
                    text = text[:3000] + "... (truncated)"

                # Extract meta description
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                description = meta_desc.get('content', '') if meta_desc else ''

                info = {
                    'title': title,
                    'description': description,
                    'content': text,
                    'url': url
                }

                return info, None
        except Exception as e:
            logger.error(f"Error scraping webpage: {e}")
            return None, f"Webpage scraping error: {str(e)}"

    async def process_media(self, url):
        """Main function to process any media URL"""
        try:
            # Download the file
            file_path, error = await self.download_file(url)
            if error:
                return None, error

            # Determine file type using mimetypes and file extension
            file_type, _ = mimetypes.guess_type(url)
            if not file_type:
                # Try to determine by extension from URL
                ext = os.path.splitext(urlparse(url).path)[1].lower()
                if ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
                    file_type = 'image/' + ext[1:]
                elif ext in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml']:
                    file_type = 'text/plain'
                else:
                    # Try to determine from downloaded file
                    with open(file_path, 'rb') as f:
                        header = f.read(8)
                        if header.startswith(b'\x89PNG'):
                            file_type = 'image/png'
                        elif header.startswith(b'\xff\xd8\xff'):
                            file_type = 'image/jpeg'
                        elif header.startswith(b'GIF8'):
                            file_type = 'image/gif'
                        else:
                            file_type = 'application/octet-stream'

            # Process based on type
            if file_type and file_type.startswith('image/'):
                result, error = await self.analyze_image(file_path)
            else:
                result, error = await self.analyze_document(file_path)

            # Clean up temp file
            try:
                os.remove(file_path)
            except:
                pass

            return result, error
        except Exception as e:
            logger.error(f"Error processing media: {e}")
            return None, f"Media processing error: {str(e)}"

    def is_valid_url(self, url):
        """Check if URL is valid"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

    def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass
