import os
from typing import Any, Optional, Dict, List
from smolagents.tools import Tool
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSearchTool(Tool):
    name = "web_search"
    description = "Performs a web search and returns the top search results."
    inputs = {'query': {'type': 'string', 'description': 'The search query to perform.'}}
    output_type = "string"

    def __init__(self, max_results: int = 10, **kwargs):
        """
        Initialize the web search tool with both DuckDuckGo and SERP API capabilities
        
        Args:
            max_results (int): Maximum number of search results to return
            serpapi_key (str, optional): SERP API key for fallback searches
            **kwargs: Additional arguments passed to DuckDuckGo search
        """
        super().__init__()
        self.max_results = max_results
        self.serpapi_key = os.getenv('SERPAPI_API_KEY')
        self.ddg_available = False
        self.serpapi_available = False
        
        # Initialize DuckDuckGo
        try:
            from duckduckgo_search import DDGS
            self.ddgs = DDGS(**kwargs)
            self.ddg_available = True
        except ImportError:
            logger.warning("DuckDuckGo search package not available. Install with: pip install duckduckgo-search")
        
        # Initialize SERP API as fallback
        if self.serpapi_key:
            try:
                from serpapi import GoogleSearch
                self.serpapi_available = True
            except ImportError:
                logger.warning("SERP API package not available. Install with: pip install google-search-results")
        else:
            logger.warning("SERP API key not provided. Fallback search will not be available.")

    def _search_with_ddg(self, query: str) -> List[Dict[str, str]]:
        """
        Perform search using DuckDuckGo
        
        Args:
            query (str): Search query
            
        Returns:
            List[Dict[str, str]]: List of search results
            
        Raises:
            Exception: If search fails or no results found
        """
        if not self.ddg_available:
            raise Exception("DuckDuckGo search not available")
            
        try:
            results = list(self.ddgs.text(query, max_results=self.max_results))
            if not results:
                raise Exception("No results found from DuckDuckGo")
            return results
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {str(e)}")
            raise

    def _search_with_serp(self, query: str) -> List[Dict[str, str]]:
        """
        Perform search using SERP API
        
        Args:
            query (str): Search query
            
        Returns:
            List[Dict[str, str]]: List of search results
            
        Raises:
            Exception: If search fails or no results found
        """
        if not self.serpapi_available or not self.serpapi_key:
            raise Exception("SERP API search not available")

        try:
            from serpapi import GoogleSearch
            search = GoogleSearch({
                "q": query,
                "api_key": self.serpapi_key,
                "num": self.max_results
            })
            results = search.get_dict().get('organic_results', [])
            
            if not results:
                raise Exception("No results found from SERP API")
            
            # Convert SERP API results to match DuckDuckGo format
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'title': result.get('title', ''),
                    'href': result.get('link', ''),
                    'body': result.get('snippet', '')
                })
            return formatted_results
        except Exception as e:
            logger.error(f"SERP API search failed: {str(e)}")
            raise

    def forward(self, query: str) -> str:
        """
        Perform web search with fallback options
        
        Args:
            query (str): Search query
            
        Returns:
            str: Formatted search results
            
        Raises:
            Exception: If both search methods fail or return no results
        """
        if not query:
            raise ValueError("Search query cannot be empty")
            
        ddg_error = None
        serp_error = None
        results = []
        
        # Try DuckDuckGo first
        if self.ddg_available:
            try:
                results = self._search_with_ddg(query)
            except Exception as e:
                ddg_error = str(e)
                logger.warning(f"DuckDuckGo search failed: {ddg_error}")
            
        # If DuckDuckGo fails, try SERP API
        if not results and self.serpapi_available and self.serpapi_key:
            logger.info("Falling back to SERP API")
            try:
                results = self._search_with_serp(query)
            except Exception as e:
                serp_error = str(e)
                logger.error(f"SERP API search failed: {serp_error}")
        
        # If both searches fail
        if not results:
            errors = []
            if ddg_error:
                errors.append(f"DuckDuckGo: {ddg_error}")
            if serp_error:
                errors.append(f"SERP API: {serp_error}")
            error_msg = "Search failed! " + " | ".join(errors)
            raise Exception(error_msg)

        # Format results
        postprocessed_results = []
        for result in results:
            if all(k in result for k in ['title', 'href', 'body']):
                postprocessed_results.append(
                    f"[{result['title']}]({result['href']})\n{result['body']}"
                )
        
        if not postprocessed_results:
            raise Exception("No valid results could be processed")
            
        return "## Search Results\n\n" + "\n\n".join(postprocessed_results)