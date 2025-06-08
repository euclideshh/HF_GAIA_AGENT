import os
from typing import Dict, List
from smolagents.tools import Tool
from langchain_community.tools import TavilySearchResults
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the web for a query using Tavily API and return the top 3 most relevant results."
    inputs = {'query': {'type': 'string', 'description': 'The search query to perform.'}}
    output_type = "object"

    def __init__(self, max_results: int = 3, **kwargs):
        """
        Initialize the web search tool with Tavily API
        """
        super().__init__()
        
        # Verify Tavily API key is set
        if not os.getenv('TAVILY_API_KEY'):
            logger.error("TAVILY_API_KEY environment variable not set")
            raise ValueError("TAVILY_API_KEY environment variable must be set")
        
        # Initialize Tavily search
        try:
            self.search_tool = TavilySearchResults(max_results=max_results)
        except Exception as e:
            logger.error(f"Failed to initialize Tavily search: {str(e)}")
            raise

    def forward(self, query: str) -> Dict[str, str]:
        """
        Search Tavily for a query and return maximum 3 results.
        
        Args:
            query (str): The search query to perform.
            
        Returns:
            Dict[str, str]: Dictionary containing formatted search results
        """
        try:
            if not query:
                raise ValueError("Search query cannot be empty")

            # Perform search using Tavily
            search_results = self.search_tool.invoke({"query": query})
            
            if not search_results:
                return {"web_results": "No search results found"}
            
            if isinstance(search_results, List):
                # Format the results
                formatted_results = "\n\n---\n\n".join(
                    [
                        f'<Document source="{result.get("url", "")}" page=""/>\n{result.get("content", "")}\n</Document>'
                        for result in search_results
                    ]
                )
            else:
                logger.warning(f"Unexpected search results format: {type(search_results)}")
                return {"web_results": "Unexpected search results format"}
            
            return {"web_results": formatted_results}
            
        except Exception as e:
            error_msg = f"Search failed: {str(e)}"
            logger.error(error_msg)
            return {"web_results": error_msg}