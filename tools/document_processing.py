from typing import Any, Optional, Dict
from smolagents.tools import Tool
import os
import tempfile
import requests
from urllib.parse import urlparse
import uuid
import pandas as pd

class DocumentProcessingTool(Tool):
    name = "document_processing"
    description = "Process various document types including saving files, downloading files, and analyzing CSV/Excel files."
    inputs = {
        'action': {'type': 'string', 'description': 'The action to perform: "save", "download", "analyze_csv", "analyze_excel"'},
        'content': {'type': 'string', 'description': 'The content to process: text content for save, URL for download, filepath for analysis'}
    }
    output_type = "string"

    def __init__(self):
        """Initialize the document processing tool"""
        super().__init__()
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("You must install required packages: pip install pandas openpyxl") from e
        
        self.is_initialized = True

    def _save_and_read_file(self, content: str) -> str:
        """
        Save content to a file and return the path.
        Args:
            content (str): the content to save to the file
        """
        temp_dir = tempfile.gettempdir()
        random_filename = f"file_{uuid.uuid4().hex[:8]}.txt"
        filepath = os.path.join(temp_dir, random_filename)

        with open(filepath, "w") as f:
            f.write(content)

        return f"File saved to {filepath}. You can read this file to process its contents."

    def _download_file_from_url(self, url: str) -> str:
        """
        Download a file from a URL and save it to a temporary location.
        Args:
            url (str): the URL of the file to download.
        """
        try:
            # Generate random filename with original extension if available
            path = urlparse(url).path
            ext = os.path.splitext(path)[1] or '.tmp'
            random_filename = f"downloaded_{uuid.uuid4().hex[:8]}{ext}"

            # Create temporary file
            temp_dir = tempfile.gettempdir()
            filepath = os.path.join(temp_dir, random_filename)

            # Download the file
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Save the file
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return f"File downloaded to {filepath}. You can read this file to process its contents."
        except Exception as e:
            return f"Error downloading file: {str(e)}"

    def _analyze_csv_file(self, file_path: str) -> str:
        """
        Analyze a CSV file using pandas.
        Args:
            file_path (str): the path to the CSV file.
        """
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Run various analyses
            result = f"CSV file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
            result += f"Columns: {', '.join(df.columns)}\n\n"

            # Add summary statistics
            result += "Summary statistics:\n"
            result += str(df.describe())
            result += "\n\nTop 50 rows content:\n"
            result += str(df.head(50))

            return result
        except Exception as e:
            return f"Error analyzing CSV file: {str(e)}"

    def _analyze_excel_file(self, file_path: str) -> str:
        """
        Analyze an Excel file using pandas.
        Args:
            file_path (str): the path to the Excel file.            
        """
        try:
            # Read the Excel file
            df = pd.read_excel(file_path)

            # Run various analyses
            result = f"Excel file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
            result += f"Columns: {', '.join(df.columns)}\n\n"

            # Add summary statistics
            result += "Summary statistics:\n"
            result += str(df.describe())
            result += "\n\nTop 50 rows content:\n"
            result += str(df.head(50))           

            return result
        except Exception as e:
            return f"Error analyzing Excel file: {str(e)}"

    def forward(self, action: str, content: str) -> str:
        """
        Process documents based on the specified action.
        
        Args:
            action (str): The action to perform (save, download, analyze_csv, analyze_excel)
            content (str): The content to process
            
        Returns:
            str: Result of the operation
        """
        try:
            if action == "save":
                return self._save_and_read_file(content)
                
            elif action == "download":
                return self._download_file_from_url(content)
                
            elif action == "analyze_csv":
                return self._analyze_csv_file(content)
                
            elif action == "analyze_excel":                
                return self._analyze_excel_file(content)
                
            else:
                if ((content.find(".xls") != -1) or (content.find(".xlsx") != -1)):
                    return self._analyze_excel_file(content)
                elif (content.find("csv") != -1):
                    return self._analyze_csv_file(content)
                else:
                    return f"Error: Invalid action '{action}'. Valid actions are: save, download, analyze_csv, analyze_excel"
                
        except Exception as e:
            return f"Error performing {action} operation: {str(e)}"
