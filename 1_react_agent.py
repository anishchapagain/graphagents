"""
This script demonstrates the use of a ReAct agent with two different
language models (OpenAI and Groq) to perform web searches using Tavily.
The agent randomly selects one of the two models for each of the six queries.
"""

import os
import random
# from dotenv import load_dotenv

from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch

# Load environment variables from .env file
# load_dotenv()


def get_openai_llm():
    """Initializes and returns the OpenAI language model."""
    return ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-3.5-turbo-0125",
        temperature=0,
    )


def get_groq_llm():
    """Initializes and returns the Groq language model."""
    return init_chat_model("groq:llama3-8b-8192")


# Search tool using Tavily
search = TavilySearch(
    api_key=os.getenv("TAVILY_API_KEY"),
    search_type="web",
    max_results=3,
)


@tool
def search_tool(query: str) -> str:
    """
    A tool that uses the TavilySearch to search the web for a query.
    """
    return search.invoke(query)


def main():
    """
    Main function to run the agent with random models and queries.
    """
    queries = [
        "What is the weather in New York?",
        "Who won the last FIFA World Cup?",
        "What is the capital of Australia?",
        "What are the latest advancements in AI?",
        "What is the recipe for a margarita?",
        "Who is the current president of the United States?",
    ]

    for query in queries:
        # Randomly select a language model
        llm_choice = random.choice(["openai", "groq"])
        if llm_choice == "openai":
            llm = get_openai_llm()
            print("--- Using OpenAI model ---")
        else:
            llm = get_groq_llm()
            print("--- Using Groq model ---")

        # Initialize the agent
        agent = initialize_agent(
            tools=[search_tool],
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
        )

        # Invoke the agent with the query
        try:
            response = agent.invoke(query)
            print(f"Query: {query}")
            print(f"Response: {response}")
        except Exception as e:
            print(f"An error occurred while processing the query '{query}': {e}")


if __name__ == "__main__":
    main()