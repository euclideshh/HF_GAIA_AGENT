from typing import Any, Optional
from smolagents.tools import Tool

class MathOperationsTool(Tool):
    name = "math_operations"
    description = "Provides a simple tools to perform math operations like multiplication, addition, subtraction, division."
    inputs = {'a': {'type': 'number', 'description': 'First value'}, 'b': {'type': 'number', 'description': 'Second value'}}
    output_type = "number"
    
    def multiply(a: float, b: float) -> float:
        """Multiply two numbers.
        Args:
            a: first int
            b: second int
        """
        return a * b

    
    def add(a: float, b: float) -> float:
        """Add two numbers.
        
        Args:
            a: first int
            b: second int
        """
        return a + b

    
    def subtract(a: float, b: float) -> float:
        """Subtract two numbers.
        
        Args:
            a: first int
            b: second int
        """
        return a - b

    
    def divide(a: float, b: float) -> float:
        """Divide two numbers.
        
        Args:
            a: first int
            b: second int
        """
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        return a / b
    
    def modulus(a: float, b: float) -> float:
        """Get the modulus of two numbers.
        
        Args:
            a: first int
            b: second int
        """
        return a % b
    
    def power(a: float, b: float) -> float:
        """Raise a number to the power of another number.
        
        Args:
            a: first int
            b: second int
        """
        return a ** b
    
    def square_root(a: float) -> float:
        """Get the square root of a number.
        
        Args:
            a: first int
        """
        if a < 0:
            raise ValueError("Cannot get square root of negative number.")
        return a ** 0.5