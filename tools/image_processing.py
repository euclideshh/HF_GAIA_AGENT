import os
import io
import base64
import uuid
from PIL import Image
from typing import Dict, Any, Optional
from smolagents.tools import Tool

class ImageProcessingTool(Tool):
    name = "image_processing"
    description = "Process and manipulate images with operations like resizing, format conversion, and base64 encoding/decoding."
    inputs = {
        'action': {'type': 'string', 'description': 'The action to perform (encode, decode, resize, rotate, convert)'},
        'content': {'type': 'string', 'description': 'The image content - either a file path or base64 string'},
        'params': {'type': 'object', 'description': 'Additional parameters for the action (e.g., size for resize)', 'nullable': True}
    }
    output_type = "object"

    def __init__(self, output_dir: str = "image_outputs"):
        super().__init__()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def encode_image(self, image_path: str) -> str:
        """Convert an image file to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def decode_image(self, base64_string: str) -> Image.Image:
        """Convert a base64 string to a PIL Image."""
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data))

    def save_image(self, image: Image.Image) -> str:
        """Save a PIL Image to disk and return the path."""
        image_id = str(uuid.uuid4())
        image_path = os.path.join(self.output_dir, f"{image_id}.png")
        image.save(image_path)
        return image_path

    def resize_image(self, image: Image.Image, size: tuple) -> Image.Image:
        """Resize an image to the specified dimensions."""
        return image.resize(size, Image.Resampling.LANCZOS)

    def rotate_image(self, image: Image.Image, degrees: float) -> Image.Image:
        """Rotate an image by the specified degrees."""
        return image.rotate(degrees, expand=True)

    def convert_format(self, image: Image.Image, format: str) -> Image.Image:
        """Convert image to specified format."""
        if image.mode != format:
            return image.convert(format)
        return image

    def forward(self, action: str, content: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process an image according to the specified action.
        
        Args:
            action: The operation to perform (encode, decode, resize, rotate, convert)
            content: The image content (file path or base64 string)
            params: Additional parameters for the action
        
        Returns:
            Dict containing the result of the operation
        """
        try:
            params = params or {}
            
            if action == "encode":
                if not os.path.exists(content):
                    return {"error": f"File not found: {content}"}
                result = self.encode_image(content)
                return {"base64_string": result}
            
            elif action == "decode":
                image = self.decode_image(content)
                path = self.save_image(image)
                return {"image_path": path}
            
            elif action in ["resize", "rotate", "convert"]:
                # First load the image
                if os.path.exists(content):
                    image = Image.open(content)
                else:
                    try:
                        image = self.decode_image(content)
                    except:
                        return {"error": "Content must be a valid file path or base64 string"}
                
                # Perform the requested operation
                if action == "resize":
                    if "size" not in params:
                        return {"error": "Size parameter required for resize"}
                    image = self.resize_image(image, tuple(params["size"]))
                
                elif action == "rotate":
                    if "degrees" not in params:
                        return {"error": "Degrees parameter required for rotate"}
                    image = self.rotate_image(image, float(params["degrees"]))
                
                elif action == "convert":
                    if "format" not in params:
                        return {"error": "Format parameter required for convert"}
                    image = self.convert_format(image, params["format"])
                
                # Save and return the result
                path = self.save_image(image)
                return {
                    "image_path": path,
                    "dimensions": image.size,
                    "format": image.format,
                    "mode": image.mode
                }
            
            else:
                return {"error": f"Unsupported action: {action}"}
                
        except Exception as e:
            return {"error": f"Error processing image: {str(e)}"}
