from typing import Any, Optional
from smolagents.tools import Tool
class FinalAnswerTool(Tool):
    name = "math_operations"
    description = "Provides a simple tools to perform math operations like multiplication, addition, subtraction, division."
    inputs = {'answer': {'type': 'any', 'description': 'The final answer to the math problem'}}
    output_type = "any"
    
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers.
        Args:
            a: first int
            b: second int
        """
        return a * b

    
    def add(a: int, b: int) -> int:
        """Add two numbers.
        
        Args:
            a: first int
            b: second int
        """
        return a + b

    
    def subtract(a: int, b: int) -> int:
        """Subtract two numbers.
        
        Args:
            a: first int
            b: second int
        """
        return a - b

    
    def divide(a: int, b: int) -> int:
        """Divide two numbers.
        
        Args:
            a: first int
            b: second int
        """
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        return a / b
