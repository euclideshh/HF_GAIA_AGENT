from typing import Any, Optional
from smolagents.tools import Tool

class MathOperationsTool(Tool):
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
    
    def modulus(a: int, b: int) -> int:
        """Get the modulus of two numbers.
        
        Args:
            a: first int
            b: second int
        """
        return a % b
    
    def power(a: int, b: int) -> int:
        """Raise a number to the power of another number.
        
        Args:
            a: first int
            b: second int
        """
        return a ** b
    
    def square_root(a: int) -> float:
        """Get the square root of a number.
        
        Args:
            a: first int
        """
        if a < 0:
            raise ValueError("Cannot get square root of negative number.")
        return a ** 0.5