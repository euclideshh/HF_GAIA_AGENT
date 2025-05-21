from typing import Any, Optional, Dict
from smolagents.tools import Tool
import os
import tempfile
import requests
import pandas as pd
import pytesseract
from PIL import Image
import tabula
from pdf2image import convert_from_path
import json
import shutil
from typing import List
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pdfplumber
import io

class DocumentProcessingTool(Tool):
    name = "document_processing"
    description = "Process various document types including PDFs, images, Excel files, and CSVs. Can download files, extract text, and analyze data."
    inputs = {
        'action': {'type': 'string', 'description': 'The action to perform: "download", "save", "ocr", "analyze_csv", "analyze_excel", "analyze_pdf"'},
        'content': {'type': 'string', 'description': 'The content to process: URL for download, file content for save, filepath for analysis'},
        'query': {'type': 'string', 'description': 'The question to answer when analyzing data files', 'nullable': True},
        'options': {'type': 'object', 'description': 'Additional options for processing (e.g., file format, sheet name)', 'nullable': True}
    }
    output_type = "string"

    def __init__(self, temp_dir="doc_processing", **kwargs):
        super().__init__()
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)
        
        # Check if running in HuggingFace space
        self.is_hf_space = bool(os.getenv("SPACE_ID"))
        
        try:
            if self.is_hf_space:
                from transformers import TrOCRProcessor, VisionEncoderDecoderModel
                self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
                self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
            import pytesseract
            import pandas as pd
            import tabula
        except ImportError as e:
            if self.is_hf_space:
                required = "transformers torch Pillow python-docx openpyxl tabula-py pdf2image pandas"
            else:
                required = "pytesseract Pillow python-docx openpyxl tabula-py pdf2image pandas"
            raise ImportError(
                f"You must install required packages: pip install {required}"
            ) from e
        
        self.is_initialized = True
    
    def _download_file(self, url: str) -> str:
        """Download a file from URL and save it to temp directory."""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get filename from URL or headers
            content_disp = response.headers.get('content-disposition')
            if content_disp and 'filename=' in content_disp:
                filename = content_disp.split('filename=')[1].strip('"')
            else:
                filename = url.split('/')[-1]
            
            filepath = os.path.join(self.temp_dir, filename)
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return filepath
        except Exception as e:
            return f"Error downloading file: {str(e)}"

    def _save_content(self, content: str, options: Optional[Dict] = None) -> str:
        """Save content to a file."""
        try:
            file_format = options.get('format', 'txt') if options else 'txt'
            filename = f"saved_content.{file_format}"
            filepath = os.path.join(self.temp_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return filepath
        except Exception as e:
            return f"Error saving content: {str(e)}"

    def _ocr_image(self, filepath: str) -> str:
        """Extract text from image using OCR."""
        try:
            image = Image.open(filepath)
            
            if self.is_hf_space:
                # Use TrOCR in HuggingFace space
                import torch
                pixel_values = self.processor(image, return_tensors="pt").pixel_values
                generated_ids = self.model.generate(pixel_values)
                text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            else:
                # Use pytesseract locally
                text = pytesseract.image_to_string(image)
            
            return text.strip()
        except Exception as e:
            return f"Error performing OCR: {str(e)}"

    def _analyze_csv(self, filepath: str, query: str) -> str:
        """Analyze CSV file and answer questions about it."""
        try:
            df = pd.read_csv(filepath)
            result = self._analyze_dataframe(df, query)
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error analyzing CSV: {str(e)}"

    def _analyze_excel(self, filepath: str, query: str, options: Optional[Dict] = None) -> str:
        """Analyze Excel file and answer questions about it."""
        try:
            sheet_name = options.get('sheet_name', 0) if options else 0
            df = pd.read_excel(filepath, sheet_name=sheet_name)
            result = self._analyze_dataframe(df, query)
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error analyzing Excel: {str(e)}"    
        
    def _analyze_pdf(self, filepath: str, query: str) -> str:
        """Analyze PDF file and answer questions about it using PDFPlumber and OCR."""
        try:
            results = {
                "text": [],
                "tables": [],
                "images": []
            }
            
            # Use PDFPlumberLoader for initial text and table extraction
            loader = PDFPlumberLoader(filepath)
            pages = loader.load()
            
            # Process each page with pdfplumber for detailed analysis
            with pdfplumber.open(filepath) as pdf:
                for i, page in enumerate(pdf.pages):
                    # Extract text
                    text = page.extract_text()
                    if text:
                        results["text"].append({
                            "page": i + 1,
                            "content": text
                        })
                    
                    # Extract tables using pdfplumber's built-in table detection
                    tables = page.extract_tables()
                    if tables:
                        for table_num, table in enumerate(tables):
                            # Convert table to DataFrame for analysis
                            df = pd.DataFrame(table[1:], columns=table[0] if table else None)
                            analysis = self._analyze_dataframe(df, query)
                            results["tables"].append({
                                "page": i + 1,
                                "table": table_num + 1,
                                "analysis": analysis
                            })
                    
                    # Extract and process images
                    images = page.images
                    if images:
                        for img_num, img in enumerate(images):
                            # Save image temporarily for OCR
                            img_bytes = io.BytesIO(img["stream"].get_data())
                            image = Image.open(img_bytes)
                              # Perform OCR on the image using the appropriate method
                            if self.is_hf_space:
                                # Use TrOCR in HuggingFace space
                                import torch
                                pixel_values = self.processor(image, return_tensors="pt").pixel_values
                                generated_ids = self.model.generate(pixel_values)
                                text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                            else:
                                # Use pytesseract locally
                                text = pytesseract.image_to_string(image)

                            if text.strip():
                                results["images"].append({
                                    "page": i + 1,
                                    "image": img_num + 1,
                                    "bbox": img["bbox"],
                                    "text": text.strip()
                                })
            
            # Split text into chunks for better processing
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            
            # Prepare final response based on query
            if "table" in query.lower():
                return json.dumps({"tables": results["tables"]}, indent=2)
            elif "image" in query.lower():
                return json.dumps({"images": results["images"]}, indent=2)
            else:
                # Return all results by default
                return json.dumps(results, indent=2)
                
        except Exception as e:
            return f"Error analyzing PDF: {str(e)}"

    def _analyze_dataframe(self, df: pd.DataFrame, query: str) -> Dict:
        """Analyze a pandas DataFrame based on the query."""
        result = {
            "shape": df.shape,
            "columns": list(df.columns),
            "summary": df.describe().to_dict(),
        }
        
        # Handle specific queries
        if "count" in query.lower():
            result["count"] = len(df)
        if "average" in query.lower() or "mean" in query.lower():
            col = [c for c in df.columns if df[c].dtype.kind in 'biufc']
            if col:
                result["means"] = df[col].mean().to_dict()
        if "sum" in query.lower():
            col = [c for c in df.columns if df[c].dtype.kind in 'biufc']
            if col:
                result["sums"] = df[col].sum().to_dict()
                
        return result

    def forward(self, action: str, content: str, query: Optional[str] = None, options: Optional[Dict] = None) -> str:
        try:
            if action == "download":
                return self._download_file(content)
                
            elif action == "save":
                return self._save_content(content, options)
                
            elif action == "ocr":
                return self._ocr_image(content)
                
            elif action == "analyze_csv":
                if not query:
                    return "Error: Query is required for CSV analysis"
                return self._analyze_csv(content, query)
                
            elif action == "analyze_excel":
                if not query:
                    return "Error: Query is required for Excel analysis"
                return self._analyze_excel(content, query, options)
                
            elif action == "analyze_pdf":
                if not query:
                    return "Error: Query is required for PDF analysis"
                return self._analyze_pdf(content, query)
                
            else:
                return f"Error: Invalid action '{action}'. Valid actions are: download, save, ocr, analyze_csv, analyze_excel, analyze_pdf"
                
        except Exception as e:
            return f"Error performing {action} operation: {str(e)}"
    
    def __del__(self):
        """Cleanup temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
