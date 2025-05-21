from typing import Any, Optional
from smolagents.tools import Tool
from langchain_community.document_loaders import ArxivLoader

class ArxivSearchTool(Tool):
    name = "arxiv_search"
    description = "Search arXiv papers based on a query and return relevant papers with their abstracts. Useful for finding scientific papers, research articles, and academic content."
    inputs = {'query': {'type': 'string', 'description': 'The search query to look up papers on arXiv.'}}
    output_type = "string"

    def __init__(self, load_max_docs=3, **kwargs):
        super().__init__()
        self.load_max_docs = load_max_docs
        try:
            import arxiv            
        except ImportError as e:
            raise ImportError(
                "You must install package `arxiv` to run this tool: run `pip install arxiv`."
            ) from e
        self.is_initialized = True

    def forward(self, query: str) -> str:
        try:            
            import pymupdf
        except ImportError as e:
            raise ImportError(
                "You must install package `pymupdf` to run this tool: run `pip install pymupdf`."
            ) from e        
        try:
            # Use ArxivLoader from langchain_community to load papers
            loader = ArxivLoader(
                query=query,
                load_max_docs=self.load_max_docs,
                load_all_available_meta=True
            )
            
            # Get the documents (papers)
            docs = loader.load()
            
            if not docs:
                return f"No arXiv papers found for the query: {query}"
            
            # Format the results nicely
            results = []
            for doc in docs:
                # Extract metadata
                metadata = doc.metadata
                title = metadata.get('Title', 'Untitled')
                authors = metadata.get('Authors', 'Unknown Authors')
                published = metadata.get('Published', 'Unknown Date')
                paper_url = metadata.get('Entry Id', '#')
                
                # Get the abstract (usually in the page_content)
                abstract = doc.page_content[:800] + "..." if len(doc.page_content) > 800 else doc.page_content
                
                # Format each paper
                paper = f"## {title}\n\n**Authors:** {authors}\n**Published:** {published}\n**URL:** {paper_url}\n\n**Abstract:**\n{abstract}\n\n---\n\n"
                results.append(paper)
            
            return "\n".join(results)
            
        except Exception as e:
            return f"Error searching arXiv: {str(e)}"
