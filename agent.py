from smolagents import CodeAgent, HfApiModel, load_tool, tool
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool
#from tools.math_operations import MathOperationsTool
from tools.visit_webpage import VisitWebpageTool
from tools.web_search import DuckDuckGoSearchTool
from tools.wikipedia_search import WikipediaSearchTool
from tools.arxiv_search import ArxivSearchTool
from tools.rag_search import RAGSearchTool

# Custom tools 
visit_webpage = VisitWebpageTool()
internet_search = DuckDuckGoSearchTool(max_results=5)
#math_tools = MathOperationsTool()
final_answer = FinalAnswerTool()
wikipedia_search = WikipediaSearchTool(load_max_docs=2)
arxiv_search = ArxivSearchTool(load_max_docs=3)
rag_search = RAGSearchTool(persist_dir="rag_db")
# Import tool from Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

model = HfApiModel(
    max_tokens=2096,
    temperature=0.5,
    model_id='Qwen/Qwen2.5-Coder-32B-Instruct',
    custom_role_conversions=None,
)

with open("prompt.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model=model,
    tools=[image_generation_tool, visit_webpage, internet_search, wikipedia_search, arxiv_search, rag_search, final_answer], ## math_tools
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)

