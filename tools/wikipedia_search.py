from typing import Any, Optional
from smolagents.tools import Tool
from langchain_community.document_loaders import WikipediaLoader

class WikipediaSearchTool(Tool):
    name = "wikipedia_search"
    description = "Search Wikipedia articles based on a query and return the content. Useful for gathering information about topics, people, places, events, etc."
    inputs = {'query': {'type': 'string', 'description': 'The search query to look up on Wikipedia.'}}
    output_type = "string"

    def __init__(self, load_max_docs=2, **kwargs):
        super().__init__()
        self.load_max_docs = load_max_docs
        try:
            import wikipedia
        except ImportError as e:
            raise ImportError(
                "You must install package `wikipedia-api` to run this tool: run `pip install wikipedia-api`."
            ) from e
        self.is_initialized = True

    def forward(self, query: str) -> str:
        try:
            # Use WikipediaLoader from langchain_community to load articles
            loader = WikipediaLoader(
                query=query,
                load_max_docs=self.load_max_docs,
                load_all_available_meta=True
            )
            
            # Get the documents (articles)
            docs = loader.load()
            
            if not docs:
                return f"No Wikipedia articles found for the query: {query}"
            
            # Format the results nicely
            results = []
            for doc in docs:
                # Extract metadata and content
                metadata = doc.metadata
                title = metadata.get('title', 'Untitled')
                summary = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                
                # Format each article
                article = f"## {title}\n\n{summary}\n\n"
                results.append(article)
            
            return "\n".join(results)
            
        except Exception as e:
            return f"Error searching Wikipedia: {str(e)}"
