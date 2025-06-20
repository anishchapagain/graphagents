from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(city:str) -> str:
    """
    Fetches the current weather information for a specified city asynchronously.

    Args:
        city (str): The name of the city for which to retrieve weather data.

    Returns:
        str: A string containing the weather information for the specified city.

    Raises:
        Exception: If there is an error retrieving the weather data.

    Example:
        weather = await get_weather("London")
    """
    
    # TODO: tavily and duckduckgo
    return "London is very cold"

if __name__ == "__main__":
    mcp.run(transport="streamable-http") # Uvicorn running on http://127.0.0.1:8000