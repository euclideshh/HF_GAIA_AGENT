from typing import Any, Optional, List, Dict
from smolagents.tools import Tool
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import json

class RAGSearchTool(Tool):
    name = "rag_search"
    description = "A RAG (Retrieval Augmented Generation) tool that can store documents and perform semantic search using FAISS and GTE-small embeddings."
    inputs = {
        'action': {'type': 'string', 'description': 'The action to perform: "add" to add documents, "search" to search existing documents, or "clear" to clear the database'},
        'content': {'type': 'string', 'description': 'For "add" action: the text content to add to the database. For "search" action: the query to search for. For "clear" action: can be empty.'},
        'metadata': {'type': 'object', 'description': 'Optional metadata for the documents when adding them', 'nullable': True}
    }
    output_type = "string"

    def __init__(self, persist_dir="rag_db", **kwargs):
        super().__init__()
        self.persist_dir = persist_dir
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import FAISS
        except ImportError as e:
            raise ImportError(
                "You must install packages `faiss-cpu` and `sentence-transformers` to run this tool: "
                "run `pip install faiss-cpu sentence-transformers`."
            ) from e
        
        # Initialize the embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="thenlper/gte-small",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
        
        # Try to load existing database or create new one
        if os.path.exists(os.path.join(persist_dir, "index.faiss")):
            self.db = FAISS.load_local(persist_dir, self.embeddings)
        else:
            self.db = None
        
        self.is_initialized = True

    def forward(self, action: str, content: str, metadata: Optional[Dict] = None) -> str:
        try:
            if action == "add":
                if not content:
                    return "Error: No content provided to add to the database."
                
                # Split text into chunks
                chunks = self.text_splitter.split_text(content)
                
                # Create metadata for each chunk if provided
                metadatas = [metadata] * len(chunks) if metadata else None
                
                # Initialize db if it doesn't exist
                if self.db is None:
                    self.db = FAISS.from_texts(chunks, self.embeddings, metadatas=metadatas)
                else:
                    self.db.add_texts(chunks, metadatas=metadatas)
                
                # Save the updated database
                self.db.save_local(self.persist_dir)
                
                return f"Successfully added {len(chunks)} chunks to the database."
                
            elif action == "search":
                if not content:
                    return "Error: No search query provided."
                if self.db is None:
                    return "Error: No documents have been added to the database yet."
                
                # Perform similarity search
                results = self.db.similarity_search_with_score(content, k=3)
                
                # Format results
                formatted_results = []
                for doc, score in results:
                    result = {
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'similarity_score': float(score)  # Convert numpy float to Python float
                    }
                    formatted_results.append(result)
                
                return json.dumps(formatted_results, indent=2)
                
            elif action == "clear":
                if os.path.exists(self.persist_dir):
                    import shutil
                    shutil.rmtree(self.persist_dir)
                self.db = None
                return "Database cleared successfully."
                
            else:
                return f"Error: Invalid action '{action}'. Valid actions are 'add', 'search', or 'clear'."
                
        except Exception as e:
            return f"Error performing {action} operation: {str(e)}"
