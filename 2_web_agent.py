# --- Documentation ---
# This script creates a ReACT agent with Tavily tools for comprehensive web research.
# The agent is designed to utilize Tavily's search, crawl, and extract capabilities 
# to gather and synthesize information from the web.
#
# Resources:
# - ReACT Agent: https://python.langchain.com/api_reference/langchain/agents/langchain.agents.react.agent.create_react_agent.html
# - Paper: https://arxiv.org/abs/2210.03629
#
# --- Imports ---
import datetime

from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch, TavilyCrawl, TavilyExtract
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage

from dotenv import load_dotenv

import os

load_dotenv()
# --- Load Environment Variables ---
# Load API keys and other environment variables from a .env file.
load_dotenv()

# --- LLM Initialization ---
def get_openai_llm():
    """Initializes and returns the OpenAI language model."""
    return ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-3.5-turbo-0125",
        temperature=0.4,
    )

def get_groq_llm():
    """Initializes and returns the Groq language model."""
    # return init_chat_model("groq:llama3-8b-8192")
    return init_chat_model("groq:llama3-8b-instant")


# Initialize the language model.
llm = get_groq_llm() # Alternative model
# llm = get_openai_llm()

# --- Tool Initialization ---
# Initialize Tavily tools for web search, extraction, and crawling.
# 1. TavilySearch: For performing web searches.
search = TavilySearch(
    api_key=os.getenv("TAVILY_API_KEY"),
    search_type="web",
    max_results=10,
    model=llm,
)

# 2. TavilyExtract: For extracting content from web pages.
extract = TavilyExtract(extract_depth="advanced", model=llm)

# 3. TavilyCrawl: For crawling websites.
crawl = TavilyCrawl()

# --- Agent Creation ---
# Get today's date for the prompt.
today = datetime.date.today().strftime("%A, %B %d, %Y")
print(f"Today: {today}")

# Create a ReACT agent with the initialized tools and a custom prompt.
web_agent = create_react_agent(
    model=llm,
    tools=[search, extract, crawl],
    prompt=ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
                You are a research agent equipped with advanced web tools: Tavily Web Search, Web Crawl, and Web Extract. 
                Your mission is to conduct comprehensive, accurate, and up-to-date research, grounding your findings in credible web sources.

                **Today's Date:** {today}

                **Available Tools:**

                1. **Tavily Web Search**
                   - **Purpose:** Retrieve relevant web pages based on a query.
                   - **Usage:** Provide a search query to receive semantically ranked results.
                   - **Best Practices:** Use specific queries and optimize with parameters like `search_depth`, `time_range`, etc.

                2. **Tavily Web Crawl**
                   - **Purpose:** Explore a website's structure and gather content from linked pages.
                   - **Usage:** Input a base URL to crawl with specified parameters.
                   - **Best Practices:** Start with shallow crawls and use path filters.

                3. **Tavily Web Extract**
                   - **Purpose:** Extract the full content from specific web pages.
                   - **Usage:** Provide URLs to retrieve detailed content.
                   - **Best Practices:** Set `extract_depth` to "advanced" for comprehensive extraction.

                **Guidelines for Conducting Research:**
                - **Citations:** Always support findings with source URLs.
                - **Accuracy:** Rely solely on data from the provided tools.
                - **Methodology:** Follow a structured Thought-Action-Observation cycle.

                **Example Workflows:**
                - **Search Only:** For quick headlines or simple questions.
                - **Search and Extract:** For detailed insights from specific articles.
                - **Search and Crawl:** For in-depth research on a topic from a single source.

                ---
                You will now receive a research question from the user:
                """
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    ),
    name="web_agent",
)

# --- Agent Invocation ---
# Define the input for the agent.
# Two example inputs are provided. The second one is active.
inputs = {
    "messages": [
        HumanMessage(
            content="find all the iphone models currently available on https://www.olizstore.com/ and their prices"
        )
    ]
}

# inputs = {
#     "messages": [
#         HumanMessage(
#             content="find top 5 jobs listed for a software engineer in jobsnepal.com"
#         )
#     ]
# }

# Stream the agent's output and print the messages.
for output in web_agent.stream(inputs, stream_mode="values"):
    messages = output["messages"][-1]
    if isinstance(messages, tuple):
        print(messages.content)
    else:
        messages.pretty_print()
