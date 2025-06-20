from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv

import asyncio

async def main():
    """
    Initialize the MultiServerMCPClient with available MCP services.

    Services:
        - mcp_math: Provides mathematical computation capabilities. Uses HTTP transport.
        - weather: Provides weather information and forecasts. Uses 'streamable' transport.
    """
    mcp_math = {
        "command": "python",
        "args":["mcp_math.py"],
        "transport": "stdio"
    }
    weather = {
        "url": "http://localhost:8000/mcp",
        "transport": "streamable_http"
    }

    # Include mcp_math and weather in a list for MultiServerMCPClient
    services = {"math":mcp_math, "weather":weather}
    client = MultiServerMCPClient(services)

    tools = await client.get_tools()
    llm  = ChatGroq(model="llama3-8b-8192")

    # create agent
    agent = create_react_agent(
        llm,
        tools
    )

    # Calling math
    response_math = await agent.invoke(
        {
            "messages":[{"role":"user", "content": "What is ouput for 4*(5+1)"}]
        }
    )

    print(f"Reponse from Math {response_math['messages'][-1].content}")

    # calling weather
    response_weather = await agent.invoke(
        {
            "messages":[{"role":"user", "content": "What is current weather in London"}]
        }
    )

    print(f"Reponse from Weather-Tavily {response_weather['messages'][-1].content}")



asyncio.run(main())