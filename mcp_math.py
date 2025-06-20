from mcp.server.fastmcp import FastMCP

# An instance of the FastMCP class initialized with the "Basic_Math" profile. 
mcp = FastMCP("Basic_Math")

# `mcp` object provides methods for performing basic mathematical operations such as addition and multiplication.
@mcp.tool()
def add(a: int | float, b: int | float) -> int | float:
    """
    Adds two numbers using Fastmcp.

    Args:
        a (int or float): The first number.
        b (int or float): The second number.

    Returns:
        int or float: The sum of a and b.
    """
    return a + b

@mcp.tool()
def multiply(a: int | float, b: int | float) -> int | float:
    """
    Multiplies two numbers using Fastmcp.

    Args:
        a (int or float): The first number.
        b (int or float): The second number.

    Returns:
        int or float: The product of a and b.
    """
    return a*b

if __name__ == "__main__":
    mcp.run(transport="stdio") # 'stdio', 'sse', 'streamable-http'