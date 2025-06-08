from smolagents import CodeAgent, InferenceClientModel, load_tool, tool #HfApiModel
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool
from tools.math_operations import MathOperationsTool
#from tools.visit_webpage import VisitWebpageTool
from tools.web_search import DuckDuckGoSearchTool
from tools.wikipedia_search import WikipediaSearchTool
from tools.arxiv_search import ArxivSearchTool
from tools.rag_search import RAGSearchTool
from tools.code_execution import CodeExecutionTool
from tools.document_processing import DocumentProcessingTool
from tools.image_processing import ImageProcessingTool
from tools.web_scraping import WebScrapingTool
from tools.youtube_processing import YouTubeVideoProcessorTool

# Instantiate the tools
#visit_webpage = VisitWebpageTool()
web_search = DuckDuckGoSearchTool(max_results=5)
math_tools = MathOperationsTool()
final_answer = FinalAnswerTool()
wikipedia_search = WikipediaSearchTool(load_max_docs=2)
arxiv_search = ArxivSearchTool(load_max_docs=3)
rag_search = RAGSearchTool(persist_dir="rag_db")
code_execution = CodeExecutionTool()
document_processing = DocumentProcessingTool(temp_dir="doc_processing")
image_generation_tool = ImageProcessingTool()
web_scraping = WebScrapingTool()
youtube_processing = YouTubeVideoProcessorTool()

# Define the tools to be used by the CodeAgent
tools = [    
    #visit_webpage, 
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

##########additional_imports = ["pandas", "numpy", "pymupdf", "bs4", "requests", "pytz", "datetime", "PIL.Image"]

# Load prompt templates
#with open("prompt.yaml", 'r') as stream:
#    prompt_templates = yaml.safe_load(stream)

# Model configuration
model = InferenceClientModel(
    max_tokens=2096,
    temperature=0,
    model_id='deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
    provider="hf-inference"
)

# CodeAgent definition
gaai_agent = CodeAgent(
    model=model,
    tools = tools, 
    verbosity_level=2,
    max_steps=15,        
    add_base_tools=True
)

#Force the agent to use the tools defined above
gaai_agent.prompt_templates["system_prompt"] = gaai_agent.prompt_templates["system_prompt"] + "\nDo not use forward method to call Tools, just use tool_name(params), example: papers = arxiv_search(query=\"transformers time series forecasting year:2023\").\nIt's important to avoid reusing code that causes errors. You have a maximum of 15 steps to answer each question, so if you cannot provide the correct answer in fewer than 15 steps, please provide your best attempt.\nIf you need to process documents (Excel, Word, PDF, etc.) or images in your task, make sure you get the correct path and file name.\nYour final answer should be a number or as few words as possible or a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."