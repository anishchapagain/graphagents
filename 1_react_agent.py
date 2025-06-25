from langchain.chat_models import init_chat_model
from langchain.agents import initialize_agent
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import os

load_dotenv()

llm=init_chat_model("groq:llama3-8b-8192")

# Search tool using Tavily
search = TavilySearch(
    api_key=os.getenv("TAVILY_API_KEY"),
    search_type="web",
    max_results=3,
    model=llm,
)
# Define a simple tool that uses the search tool
from langchain_core.tools import tool
@tool
def search_tool(query: str) -> str:
    """Search the web for a query."""
    results = search.invoke(query)
    return results

# # TEST
# result = llm.invoke("Tell me about country Iran, its location, people, and culture.")
# print(result.content)

agent = initialize_agent(tools=[search_tool], llm=llm, agent="zero-shot-react-description", verbose=True)
# Be careful with the query, as it will be sent to the Tavily API
# and may incur costs depending on your usage plan.
# Plus an infinite loop may occur if the query is not specific enough.

# TEST
agent.invoke("Tell me about country Iran, its location, people, and culture.")