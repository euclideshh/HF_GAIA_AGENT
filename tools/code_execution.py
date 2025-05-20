import os
import io
import sys
import uuid
import base64
import traceback
import contextlib
import tempfile
import subprocess
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from smolagents.tools import Tool

class CodeExecutionTool(Tool):
    name = "code_execution"
    description = "Execute Python code with support for data analysis, plotting, and file operations. Returns text output and base64-encoded images for plots."
    inputs = {
        'code': {'type': 'string', 'description': 'The Python code to execute'},
        'input_data': {'type': 'object', 'description': 'Optional input data for the code execution', 'nullable': True}
    }
    output_type = "object"

    def __init__(self, work_dir="code_execution"):
        super().__init__()
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)
        self._setup_matplotlib()
        
    def _setup_matplotlib(self):
        """Configure matplotlib for non-interactive backend"""
        plt.switch_backend('Agg')
    
    def _capture_output(self, code: str) -> Dict[str, Any]:
        """Execute code and capture output, including stdout, stderr, and plots"""
        # Create string buffers for stdout and stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        result = {
            'output': '',
            'error': '',
            'images': [],
            'dataframes': []
        }
        
        # Create a temporary namespace for code execution
        namespace = {
            'np': np,
            'pd': pd,
            'plt': plt,
            'Image': Image,
        }
        
        try:
            # Execute in controlled environment
            with contextlib.redirect_stdout(stdout_buffer), \
                 contextlib.redirect_stderr(stderr_buffer):
                
                # Execute the code
                exec(code, namespace)
                
                # Capture any active plots
                if plt.get_fignums():
                    for fig_num in plt.get_fignums():
                        fig = plt.figure(fig_num)
                        img_buffer = io.BytesIO()
                        fig.savefig(img_buffer, format='png')
                        img_buffer.seek(0)
                        img_data = base64.b64encode(img_buffer.getvalue()).decode()
                        result['images'].append(img_data)
                        plt.close(fig)
                
                # Capture any DataFrames in the namespace
                for var_name, var_value in namespace.items():
                    if isinstance(var_value, pd.DataFrame):
                        result['dataframes'].append({
                            'name': var_name,
                            'data': var_value.to_dict(orient='records')
                        })
            
            # Get output from buffers
            result['output'] = stdout_buffer.getvalue()
            result['error'] = stderr_buffer.getvalue()
            
        except Exception as e:
            result['error'] = f"Execution error: {str(e)}\n{traceback.format_exc()}"
        
        finally:
            stdout_buffer.close()
            stderr_buffer.close()
        
        return result

    def forward(self, code: str, input_data: Optional[Dict] = None) -> Dict[str, Any]:
        if not code:
            return {"error": "Error: No code provided to execute."}
        
        # If input_data is provided, add it to the setup code
        setup_code = ""
        if input_data:
            for var_name, var_value in input_data.items():
                if isinstance(var_value, (str, int, float, list, dict)):
                    setup_code += f"{var_name} = {repr(var_value)}\n"
        
        # Combine setup code with user code
        full_code = setup_code + code
        
        try:
            # Execute the code and capture all output
            result = self._capture_output(full_code)
            return result
        except Exception as e:
            return {
                "error": f"Error executing code: {str(e)}",
                "output": "",
                "images": [],
                "dataframes": []
            }

    def __del__(self):
        """Cleanup any temporary files"""
        try:
            if os.path.exists(self.work_dir):
                for root, dirs, files in os.walk(self.work_dir, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
                os.rmdir(self.work_dir)
        except Exception:
            pass  # Ignore cleanup errors
