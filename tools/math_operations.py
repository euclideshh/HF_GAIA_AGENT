from typing import Any, Optional
from smolagents.tools import Tool

class MathOperationsTool(Tool):
    name = "math_operations"
    description = "Performs mathematical operations like multiplication, addition, subtraction, division, modulus, power, and square root"
    inputs = {
        'operation': {'type': 'string', 'description': 'The operation to perform: "multiply", "add", "subtract", "divide", "modulus", "power", "square_root"'},
        'a': {'type': 'number', 'description': 'First value'},
        'b': {'type': 'number', 'description': 'Second value (not used for square_root)', 'nullable': True}
    }
    output_type = "number"
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return a + b
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract two numbers."""
        return a - b
    
    def divide(self, a: float, b: float) -> float:
        """Divide two numbers."""
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        return a / b
    
    def modulus(self, a: float, b: float) -> float:
        """Get the modulus of two numbers."""
        if b == 0:
            raise ValueError("Cannot calculate modulus with zero.")
        return a % b
    
    def power(self, a: float, b: float) -> float:
        """Raise a number to the power of another number."""
        return a ** b
    
    def square_root(self, a: float) -> float:
        """Get the square root of a number."""
        if a < 0:
            raise ValueError("Cannot get square root of negative number.")
        return a ** 0.5

    def forward(self, operation: str, a: float, b: Optional[float] = None) -> float:
        """
        Execute the requested mathematical operation.
        
        Args:
            operation: The operation to perform (multiply, add, subtract, divide, modulus, power, square_root)
            a: First number
            b: Second number (not required for square_root)
        
        Returns:
            float: Result of the mathematical operation
        """
        operations = {
            'multiply': self.multiply,
            'add': self.add,
            'subtract': self.subtract,
            'divide': self.divide,
            'modulus': self.modulus,
            'power': self.power,
            'square_root': self.square_root
        }
        
        if operation not in operations:
            raise ValueError(f"Invalid operation. Must be one of: {', '.join(operations.keys())}")
        
        if operation == 'square_root':
            return operations[operation](a)
        
        if b is None:
            raise ValueError(f"Second number (b) is required for operation: {operation}")
        
        return operations[operation](a, b)