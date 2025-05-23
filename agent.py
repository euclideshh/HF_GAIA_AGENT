from smolagents import CodeAgent, HfApiModel, load_tool, tool
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool
from tools.math_operations import MathOperationsTool
from tools.visit_webpage import VisitWebpageTool
from tools.web_search import DuckDuckGoSearchTool
from tools.wikipedia_search import WikipediaSearchTool
from tools.arxiv_search import ArxivSearchTool
from tools.rag_search import RAGSearchTool
from tools.code_execution import CodeExecutionTool
from tools.document_processing import DocumentProcessingTool
from tools.image_processing import ImageProcessingTool
from tools.web_scraping import WebScrapingTool

# Custom tools 
visit_webpage = VisitWebpageTool()
internet_search = DuckDuckGoSearchTool(max_results=5)
math_tools = MathOperationsTool()
final_answer = FinalAnswerTool()
wikipedia_search = WikipediaSearchTool(load_max_docs=2)
arxiv_search = ArxivSearchTool(load_max_docs=3)
rag_search = RAGSearchTool(persist_dir="rag_db")
code_execution = CodeExecutionTool()
document_processing = DocumentProcessingTool(temp_dir="doc_processing")
image_generation_tool = ImageProcessingTool()
web_scraping = WebScrapingTool()

# Import tool from Hub
#image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

model = HfApiModel(
    max_tokens=2096,
    temperature=0,
    model_id='Qwen/Qwen2.5-Coder-32B-Instruct',
    custom_role_conversions=None,
)

with open("prompt.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

tools = [    
    visit_webpage, 
    internet_search, 
    math_tools,
    wikipedia_search, 
    arxiv_search,
    rag_search,
    web_scraping,
    code_execution, 
    document_processing, 
    image_generation_tool,
    final_answer
]

additional_imports = [
    # Document processing
    "pymupdf",                              # PDF reading
    # HTTP & URLs
    "requests",                             # HTTP client for REST calls :contentReference[oaicite:0]{index=0}
    "urllib.parse",                         # URL parsing and construction :contentReference[oaicite:1]{index=1}

    # Data formats
    "json",                                 # JSON serialization/deserialization
    "csv",                                  # CSV reading/writing
    "xml.etree.ElementTree",                # XML parsing
    "bs4",                                  # BeautifulSoup for HTML parsing

    # Text processing
    "re",                                   # Regular expressions

    # File & OS
    "os",                                   # OS interactions (env vars, paths)
    "sys",                                  # Interpreter info
    "pathlib",                              # Object‑oriented filesystem paths
    "subprocess",                           # Safe subprocess calls

    # Computation
    "math",                                 # Advanced math functions
    "random",                               # Random sampling and shuffling

    # Date & time
    "datetime",                             # Date/time parsing and arithmetic

    # Data analysis
    "numpy",                                # Numerical arrays :contentReference[oaicite:2]{index=2}
    "pandas",                               # Tabular data manipulation

    # Imaging
    "PIL.Image",                            # Image loading/inspection

    # Logging
    "logging",                              # Structured debug/info logging
]
    
agent = CodeAgent(
    model=model,
    tools = tools, 
    additional_authorized_imports = additional_imports,
    prompt_templates=prompt_templates,
    verbosity_level=0,
)

