from typing import Any, Optional
from smolagents.tools import Tool
from langchain_community.document_loaders import ArxivLoader
#import logging

# Configurar el logger  
#logger = logging.getLogger("smolagent")
#logger.setLevel(logging.INFO)
#if not logger.handlers:
#    # Crear un handler para archivo
#    file_handler = logging.FileHandler("agent_tools_logs.txt")
#    file_handler.setLevel(logging.INFO)    
#    # Formato del log
#    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#    file_handler.setFormatter(formatter)
#    # Agregar el handler al logger
#    logger.addHandler(file_handler)
            
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
        #logger.info(f"ArxivSearchTool invocado con query: {query}")
        try:    
            #logger.info("Check if pymupdf y fitz is installed...")        
            import pymupdf
            import fitz
            #logger.info(f"Versión de fitz (PyMuPDF): {fitz.__doc__}")
            #logger.info(f"Ubicación del módulo fitz: {fitz.__file__}")
        except ImportError as e:
            raise ImportError(
                "You must install package `pymupdf` to run this tool: run `pip install pymupdf`."
            ) from e        
        try:
            # Use ArxivLoader from langchain_community to load papers
            #logger.info("Creating ArxivLoader object...")     
            loader = ArxivLoader(
                query=query,
                load_max_docs=self.load_max_docs,
                load_all_available_meta=True
            )
            
            # Get the documents (papers)
            #logger.info("ArxivLoader method load is invoked...")     
            docs = loader.load()
            
            if not docs:
                return f"No arXiv papers found for the query: {query}"
            
            # Format the results nicely
            results = []
            #logger.info("Papers found, formatting results...")     
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
            #logger.info("Formatting results ended... SUCCESS!")     
            return "\n".join(results)
            
        except Exception as e:
            return f"Error searching arXiv: {str(e)}"
