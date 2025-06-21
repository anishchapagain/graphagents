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

    import os

    tools = await client.get_tools()
    llm  = ChatGroq(model="llama3-8b-8192")

    # create agent
    agent = create_react_agent(
        llm,
        tools
    )

    # Calling math
    response_math = await agent.ainvoke(
        {
            "messages":[{"role":"user", "content": "What is output for 4 multiplied by (5+1)"}]
        }
    )

    print(f"Reponse from Math {response_math['messages'][-1].content}")

    # calling weather
    response_weather = await agent.ainvoke(
        {
            "messages":[{"role":"user", "content": "What is current weather in London"}]
        }
    )

    print(f"Reponse from Weather-Tavily {response_weather['messages'][-1].content}")



asyncio.run(main())

# Issues:1
# groq.BadRequestError: Error code: 400 - {'error': {'message': "Failed to call a function. Please adjust your prompt. See 'failed_generation' for more details.", 'type': 'invalid_request_error', 'code': 'tool_use_failed', 'failed_generation': '<tool-use>{"tool_calls":[{"id":"pending","type":"function","function":{"name":"add"},"parameters":{"a":5,"b":1}},{"id":"pending","type":"function","function":{"name":"multiply"},"parameters":{"a":4,"result":6}}]}</tool-use>'}}
# During task with name 'agent' and id 'fdbc56cf-4a0d-6768-70ea-3a6ecb662926'

# Issues:2
# Reponse from Math <tool-use>{"tool_calls":[]}</tool-use>
# Reponse from Weather-Tavily <tool-use>{"tool_calls":[]}</tool-use>

# Answers:
# Reponse from Math The answer is 24!  
# Reponse from Weather-Tavily London is very cold.

# Prompt Failed: # <tooluse> math: is not recognizing * and times
# What is output for 4 times by (5+1)"  
# What is output for 4 * (5+1)"