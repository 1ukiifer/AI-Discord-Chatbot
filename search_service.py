"""
Search Service module for Discord AI Bot
Handles web search functionality with multiple search engines
"""

import logging
import aiohttp
from typing import List, Dict, Optional
from config import Config

logger = logging.getLogger(__name__)

class SearchService:
    """Web search service supporting multiple search engines"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
                headers={'User-Agent': 'Discord-AI-Bot/1.0'}
            )
        return self.session
    def _clean_query(self, query: str) -> str:
        """Clean and optimize search query"""
        # Remove bot mentions and common Discord formatting
        query = query.replace('@', '').replace('#', '')

        # Extract key terms for better search
        stop_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'please', 'can', 'you', 'help', 'me']
        words = query.lower().split()
        important_words = [word for word in words if word not in stop_words and len(word) > 2]

        # If we have important words, use them; otherwise use original
        if important_words and len(important_words) < len(words):
            return ' '.join(important_words[:6])  # Limit to 6 key terms

        return query[:100]  # Limit query length

    def _filter_relevant_results(self, results: List[Dict], original_query: str) -> List[Dict]:
        """Filter search results for relevance"""
        if not results:
            return results

        query_words = set(original_query.lower().split())
        relevant_results = []

        for result in results:
            title = result.get('title', '').lower()
            snippet = result.get('snippet', '').lower()

            # Check if result contains query terms
            title_words = set(title.split())
            snippet_words = set(snippet.split())

            # Calculate relevance score
            title_matches = len(query_words.intersection(title_words))
            snippet_matches = len(query_words.intersection(snippet_words))

            # Require at least one match in title or snippet
            if title_matches > 0 or snippet_matches > 0:
                relevant_results.append(result)

        return relevant_results[:5]  # Return top 5 most relevant

    
    async def search(self, query: str, max_results: int = None) -> List[Dict]:
        """Main search method that tries multiple engines"""
        if not Config.ENABLE_WEB_SEARCH:
            return []

        max_results = max_results or Config.MAX_SEARCH_RESULTS

        # Clean and optimize query
        cleaned_query = self._clean_query(query)

        # Define search engines in order of preference
        search_engines = [
            ('serper', self._search_serper),  # Move Serper first - usually better results
            ('google', self._search_google),
            ('bing', self._search_bing),
            ('newsapi', self._search_newsapi),
            ('duckduckgo', self._search_duckduckgo),
        ]

        # Rest of the method stays the same...
        for engine_name, search_func in search_engines:
            try:
                logger.info(f"Trying {engine_name} search for: {cleaned_query[:100]}...")
                results = await search_func(cleaned_query, max_results)
                if results:
                    # Filter results for relevance
                    filtered_results = self._filter_relevant_results(results, query)
                    if filtered_results:
                        logger.info(f"Got {len(filtered_results)} relevant results from {engine_name}")
                        return filtered_results
            except Exception as e:
                logger.warning(f"{engine_name} search failed: {e}")
                continue

        logger.warning("All search engines failed or returned no relevant results")
        return []
    
    async def _search_google(self, query: str, max_results: int) -> List[Dict]:
        """Search using Google Custom Search API"""
        if not all([Config.GOOGLE_API_KEY, Config.GOOGLE_SEARCH_ENGINE_ID]):
            raise ValueError("Google search not configured")
        
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': Config.GOOGLE_API_KEY,
            'cx': Config.GOOGLE_SEARCH_ENGINE_ID,
            'q': query,
            'num': min(max_results, 10)
        }
        
        session = await self.get_session()
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return [
                    {
                        'title': item.get('title', ''),
                        'snippet': item.get('snippet', ''),
                        'url': item.get('link', ''),
                        'source': 'Google'
                    }
                    for item in data.get('items', [])
                ][:max_results]
            else:
                error_text = await response.text()
                raise Exception(f"Google search failed: {response.status} - {error_text}")
    
    async def _search_serper(self, query: str, max_results: int) -> List[Dict]:
        """Search using Serper API"""
        if not Config.SERPER_API_KEY:
            raise ValueError("Serper API not configured")
        
        url = "https://google.serper.dev/search"
        headers = {
            'X-API-KEY': Config.SERPER_API_KEY,
            'Content-Type': 'application/json'
        }
        payload = {'q': query, 'num': max_results}
        
        session = await self.get_session()
        async with session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return [
                    {
                        'title': item.get('title', ''),
                        'snippet': item.get('snippet', ''),
                        'url': item.get('link', ''),
                        'source': 'Serper'
                    }
                    for item in data.get('organic', [])
                ][:max_results]
            else:
                error_text = await response.text()
                raise Exception(f"Serper search failed: {response.status} - {error_text}")
    
    async def _search_bing(self, query: str, max_results: int) -> List[Dict]:
        """Search using Bing Search API"""
        if not Config.BING_API_KEY:
            raise ValueError("Bing search not configured")
        
        url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {'Ocp-Apim-Subscription-Key': Config.BING_API_KEY}
        params = {
            'q': query,
            'count': max_results,
            'textDecorations': False,
            'textFormat': 'Raw'
        }
        
        session = await self.get_session()
        async with session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return [
                    {
                        'title': item.get('name', ''),
                        'snippet': item.get('snippet', ''),
                        'url': item.get('url', ''),
                        'source': 'Bing'
                    }
                    for item in data.get('webPages', {}).get('value', [])
                ][:max_results]
            else:
                error_text = await response.text()
                raise Exception(f"Bing search failed: {response.status} - {error_text}")
    
    async def _search_duckduckgo(self, query: str, max_results: int) -> List[Dict]:
        """Search using DuckDuckGo Instant Answer API"""
        url = "https://api.duckduckgo.com/"
        params = {
            'q': query,
            'format': 'json',
            'no_html': '1',
            'skip_disambig': '1'
        }
        
        session = await self.get_session()
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                results = []
                
                # Get instant answer
                if data.get('AbstractText'):
                    results.append({
                        'title': data.get('AbstractSource', 'DuckDuckGo'),
                        'snippet': data.get('AbstractText', ''),
                        'url': data.get('AbstractURL', ''),
                        'source': 'DuckDuckGo'
                    })
                
                # Get related topics
                for topic in data.get('RelatedTopics', [])[:max_results-len(results)]:
                    if isinstance(topic, dict) and topic.get('Text'):
                        results.append({
                            'title': topic.get('FirstURL', '').split('/')[-1].replace('_', ' ') or 'Related Topic',
                            'snippet': topic.get('Text', ''),
                            'url': topic.get('FirstURL', ''),
                            'source': 'DuckDuckGo'
                        })
                
                return results[:max_results]
            else:
                raise Exception(f"DuckDuckGo search failed with status {response.status}")
    
    async def _search_newsapi(self, query: str, max_results: int) -> List[Dict]:
        """Search using NewsAPI"""
        if not Config.NEWSAPI_KEY:
            raise ValueError("NewsAPI not configured")
        
        url = "https://newsapi.org/v2/everything"
        headers = {'X-API-Key': Config.NEWSAPI_KEY}
        params = {
            'q': query,
            'sortBy': 'publishedAt',
            'pageSize': max_results,
            'language': 'en'
        }
        
        session = await self.get_session()
        async with session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return [
                    {
                        'title': article.get('title', ''),
                        'snippet': article.get('description', ''),
                        'url': article.get('url', ''),
                        'source': f"NewsAPI ({article.get('source', {}).get('name', 'Unknown')})"
                    }
                    for article in data.get('articles', [])
                ][:max_results]
            else:
                error_text = await response.text()
                raise Exception(f"NewsAPI search failed: {response.status} - {error_text}")
    
    async def close(self):
        """Close the aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()
