import asyncio
import datetime
import logging
from typing import List, Optional, Any, Dict
import requests
import pytz
import yaml
from smolagents import CodeAgent, InferenceClientModel, load_tool, tool
from concurrent.futures import TimeoutError as ConcurrentTimeoutError

# Import all your tools
from tools.final_answer import FinalAnswerTool
from tools.math_operations import MathOperationsTool
from tools.web_search import WebSearchTool
from tools.wikipedia_search import WikipediaSearchTool
from tools.arxiv_search import ArxivSearchTool
from tools.rag_search import RAGSearchTool
from tools.code_execution import CodeExecutionTool
from tools.document_processing import DocumentProcessingTool
from tools.image_processing import ImageProcessingTool
from tools.web_scraping import WebScrapingTool
from tools.youtube_processing import YouTubeVideoProcessorTool


class AsyncCodeAgentManager:
    """
    Async CodeAgent manager with timeout handling and fallback model support.
    """
    
    def __init__(
        self,
        primary_model_id: str = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
        fallback_model_id: str = 'Qwen/Qwen2.5-Coder-32B-Instruct',
        timeout_seconds: int = 120,
        max_tokens: int = 2096,
        temperature: float = 0,
        max_steps: int = 15,
        verbosity_level: int = 2,
        provider: str = "hf-inference",
        rag_persist_dir: str = "rag_db",
        doc_processing_temp_dir: str = "doc_processing",
        web_search_max_results: int = 5,
        wikipedia_max_docs: int = 2,
        arxiv_max_docs: int = 3
    ):
        """
        Initialize the AsyncCodeAgentManager.
        
        Args:
            primary_model_id: Primary model to use
            fallback_model_id: Fallback model in case of timeout
            timeout_seconds: Timeout for model responses
            max_tokens: Maximum tokens for model response
            temperature: Model temperature
            max_steps: Maximum steps for agent execution
            verbosity_level: Agent verbosity level
            provider: Model provider
            rag_persist_dir: Directory for RAG persistence
            doc_processing_temp_dir: Temporary directory for document processing
            web_search_max_results: Maximum web search results
            wikipedia_max_docs: Maximum Wikipedia documents to load
            arxiv_max_docs: Maximum ArXiv documents to load
        """
        self.primary_model_id = primary_model_id
        self.fallback_model_id = fallback_model_id
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_steps = max_steps
        self.verbosity_level = verbosity_level
        self.provider = provider
        
        # Tool configuration
        self.rag_persist_dir = rag_persist_dir
        self.doc_processing_temp_dir = doc_processing_temp_dir
        self.web_search_max_results = web_search_max_results
        self.wikipedia_max_docs = wikipedia_max_docs
        self.arxiv_max_docs = arxiv_max_docs
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize models and agents
        self.primary_agent = None
        self.fallback_agent = None
        self.tools = []
        
        # Initialize components
        self._initialize_tools()
        self._initialize_agents()
    
    def _initialize_tools(self) -> None:
        """Initialize all tools for the agent."""
        try:
            # Instantiate the tools
            web_search = WebSearchTool(max_results=self.web_search_max_results) #Corrected Tool Name
            math_tools = MathOperationsTool()
            final_answer = FinalAnswerTool()
            wikipedia_search = WikipediaSearchTool(load_max_docs=self.wikipedia_max_docs)
            arxiv_search = ArxivSearchTool(load_max_docs=self.arxiv_max_docs)
            rag_search = RAGSearchTool(persist_dir=self.rag_persist_dir)
            code_execution = CodeExecutionTool()
            document_processing = DocumentProcessingTool(temp_dir=self.doc_processing_temp_dir)
            image_generation_tool = ImageProcessingTool()
            web_scraping = WebScrapingTool()
            youtube_processing = YouTubeVideoProcessorTool()
            
            # Define the tools to be used by the CodeAgent
            self.tools = [    
                web_search, 
                math_tools,
                wikipedia_search, 
                arxiv_search,
                rag_search,
                web_scraping,
                code_execution, 
                document_processing, 
                image_generation_tool,
                youtube_processing,
                final_answer
            ]
            
            self.logger.info(f"Initialized {len(self.tools)} tools successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing tools: {str(e)}")
            raise
    
    def _create_model(self, model_id: str) -> InferenceClientModel:
        """Create a model instance with given model ID."""
        return InferenceClientModel(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            model_id=model_id,
            provider=self.provider
        )
    
    def _create_agent(self, model: InferenceClientModel) -> CodeAgent:
        """Create a CodeAgent with the given model."""
        agent = CodeAgent(
            model=model,
            tools=self.tools, 
            verbosity_level=self.verbosity_level,
            max_steps=self.max_steps,
        )
        
        # Customize the system prompt
        custom_prompt_addition = (
            "\nDo not use forward method to call Tools, just use tool_name(params), "
            "example: papers = arxiv_search(query=\"transformers time series forecasting year:2023\")."
            "\nIt's important to avoid reusing code that causes errors. You have a maximum of "
            f"{self.max_steps} steps to answer each question, so if you cannot provide the correct "
            "answer in fewer than 15 steps, please provide your best attempt."
            "\nIf you need to process documents (Excel, Word, PDF, etc.) or images in your task, "
            "make sure you get the correct path and file name."
            "\nYour final answer should be a number or as few words as possible or a comma "
            "separated list of numbers and/or strings. If you are asked for a number, don't use "
            "comma to write your number neither use units such as $ or percent sign unless "
            "specified otherwise. If you are asked for a string, don't use articles, neither "
            "abbreviations (e.g. for cities), and write the digits in plain text unless specified "
            "otherwise. If you are asked for a comma separated list, apply the above rules "
            "depending of whether the element to be put in the list is a number or a string."
        )
        
        agent.prompt_templates["system_prompt"] = (
            agent.prompt_templates["system_prompt"] + custom_prompt_addition
        )
        
        return agent
    
    def _initialize_agents(self) -> None:
        """Initialize primary and fallback agents."""
        try:
            # Create primary model and agent
            primary_model = self._create_model(self.primary_model_id)
            self.primary_agent = self._create_agent(primary_model)
            
            # Create fallback model and agent
            fallback_model = self._create_model(self.fallback_model_id)
            self.fallback_agent = self._create_agent(fallback_model)
            
            self.logger.info("Agents initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing agents: {str(e)}")
            raise
    
    async def _run_agent_with_timeout(
        self, 
        agent: CodeAgent, 
        query: str, 
        model_name: str
    ) -> Dict[str, Any]:
        """
        Run agent with timeout handling.
        
        Args:
            agent: The CodeAgent to run
            query: Query to process
            model_name: Name of the model (for logging)
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            self.logger.info(f"Running query with {model_name} model")
            
            # Run the agent in a separate thread to handle blocking operations
            loop = asyncio.get_event_loop()
            
            # Use asyncio.wait_for to implement timeout
            response = await asyncio.wait_for(
                loop.run_in_executor(None, agent.run, query),
                timeout=self.timeout_seconds
            )
            
            self.logger.info(f"Successfully completed query with {model_name} model")
            
            return {
                "response": response,
                "model_used": model_name,
                "success": True,
                "error": None,
                "execution_time": None  # Could be implemented if needed
            }
            
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout occurred with {model_name} model after {self.timeout_seconds} seconds")
            return {
                "response": None,
                "model_used": model_name,
                "success": False,
                "error": "timeout",
                "execution_time": self.timeout_seconds
            }
            
        except Exception as e:
            self.logger.error(f"Error running {model_name} model: {str(e)}")
            return {
                "response": None,
                "model_used": model_name,
                "success": False,
                "error": str(e),
                "execution_time": None
            }
    
    async def run_query(self, query: str) -> Dict[str, Any]:
        """
        Run a query with primary model and fallback to secondary model on timeout.
        
        Args:
            query: The query to process
            
        Returns:
            Dictionary containing response and execution metadata
        """
        start_time = datetime.datetime.now()
        
        # Try primary model first
        result = await self._run_agent_with_timeout(
            self.primary_agent, 
            query, 
            self.primary_model_id
        )
        
        # If primary model failed due to timeout, try fallback model
        if not result["success"] and result["error"] == "timeout":
            self.logger.info("Primary model timed out, trying fallback model")
            
            result = await self._run_agent_with_timeout(
                self.fallback_agent, 
                query, 
                self.fallback_model_id
            )
            
            if result["success"]:
                self.logger.info("Fallback model completed successfully")
            else:
                self.logger.error("Both primary and fallback models failed")
        
        # Add total execution time
        end_time = datetime.datetime.now()
        result["total_execution_time"] = (end_time - start_time).total_seconds()
        
        return result
    
    async def run_multiple_queries(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Run multiple queries concurrently.
        
        Args:
            queries: List of queries to process
            
        Returns:
            List of results for each query
        """
        tasks = [self.run_query(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions that occurred
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "response": None,
                    "model_used": None,
                    "success": False,
                    "error": str(result),
                    "query_index": i,
                    "total_execution_time": 0
                })
            else:
                result["query_index"] = i
                processed_results.append(result)
        
        return processed_results
    
    def update_timeout(self, new_timeout: int) -> None:
        """Update the timeout value."""
        self.timeout_seconds = new_timeout
        self.logger.info(f"Timeout updated to {new_timeout} seconds")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the agent manager."""
        return {
            "primary_model": self.primary_model_id,
            "fallback_model": self.fallback_model_id,
            "timeout_seconds": self.timeout_seconds,
            "max_steps": self.max_steps,
            "tools_count": len(self.tools),
            "agents_initialized": self.primary_agent is not None and self.fallback_agent is not None
        }

