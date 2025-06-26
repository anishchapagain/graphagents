# https://python.langchain.com/api_reference/langchain/agents/langchain.agents.react.agent.create_react_agent.html
# https://arxiv.org/abs/2210.03629

from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain_tavily import TavilyCrawl
from langchain_tavily import TavilyExtract

from dotenv import load_dotenv
import os

load_dotenv()

llm=init_chat_model("groq:llama3-8b-8192")

# Search tool using Tavily
search = TavilySearch(
    api_key=os.getenv("TAVILY_API_KEY"),
    search_type="web",
    max_results=10,
    model=llm,
)
# Extract tool using Tavily
extract = TavilyExtract(extract_depth="advanced", model=llm)

# Crawl tool using Tavily
crawl = TavilyCrawl()

import datetime
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # PromptTemplate

today = datetime.date.today().strftime("%A, %B %d, %Y")

# Create a ReACT agent with Tavily tools
# This agent is designed for comprehensive web research, utilizing Tavily's search, crawl, and extract capabilities to gather and synthesize information from the web.

web_agent = create_react_agent(
    model=llm,
    tools=[search, extract, crawl],
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
                f"""    
        You are a research agent equipped with advanced web tools: Tavily Web Search, Web Crawl, and Web Extract. Your mission is to conduct comprehensive, accurate, and up-to-date research, grounding your findings in credible web sources.

        **Today's Date:** {today}

        **Available Tools:**

        1. **Tavily Web Search**

        * **Purpose:** Retrieve relevant web pages based on a query.
        * **Usage:** Provide a search query to receive semantically ranked results, each containing the title, URL, and a content snippet.
        * **Best Practices:**

            * Use specific queries to narrow down results.
            * Optimize searches using parameters such as `search_depth`, `time_range`, `include_domains`, and `include_raw_content`.
            * Break down complex queries into specific, focused sub-queries.

        2. **Tavily Web Crawl**

        * **Purpose:** Explore a website's structure and gather content from linked pages for deep research and information discovery from a single source.
        * **Usage:** Input a base URL to crawl, specifying parameters such as `max_depth`, `max_breadth`, and `extract_depth`.
        * **Best Practices:**

            * Begin with shallow crawls and progressively increase depth.
            * Utilize `select_paths` or `exclude_paths` to focus the crawl.
            * Set `extract_depth` to "advanced" for comprehensive extraction.

        3. **Tavily Web Extract**

        * **Purpose:** Extract the full content from specific web pages.
        * **Usage:** Provide URLs to retrieve detailed content.
        * **Best Practices:**

            * Set `extract_depth` to "advanced" for detailed content, including tables and embedded media.
            * Enable `include_images` if image data is necessary.

        **Guidelines for Conducting Research:**

        * **Citations:** Always support findings with source URLs, clearly provided as in-text citations.
        * **Accuracy:** Rely solely on data obtained via provided tools—never fabricate information.
        * **Methodology:** Follow a structured approach:

        * **Thought:** Consider necessary information and next steps.
        * **Action:** Select and execute appropriate tools.
        * **Observation:** Analyze obtained results.
        * Repeat Thought/Action/Observation cycles as needed.
        * **Final Answer:** Synthesize and present findings with citations in markdown format.

        **Example Workflows:**

        **Workflow 1: Search Only**

        **Question:** What are recent news headlines about artificial intelligence?

        * **Thought:** I need quick, recent articles about AI.
        * **Action:** Use Tavily Web Search with the query "recent artificial intelligence news" and set `time_range` to "week".
        * **Observation:** Retrieved 10 relevant articles from reputable news sources.
        * **Final Answer:** Summarize key headlines with citations.

        **Workflow 2: Search and Extract**

        **Question:** Provide detailed insights into recent advancements in quantum computing.

        * **Thought:** I should find recent detailed articles first.
        * **Action:** Use Tavily Web Search with the query "recent advancements in quantum computing" and set `time_range` to "month".
        * **Observation:** Retrieved 10 relevant results.
        * **Thought:** I should extract content from the most comprehensive article.
        * **Action:** Use Tavily Web Extract on the most relevant URL from search results.
        * **Observation:** Extracted detailed information about quantum computing advancements.
        * **Final Answer:** Provide detailed insights summarized from extracted content with citations.

        **Workflow 3: Search and Crawl**

        **Question:** What are the latest advancements in renewable energy technologies?

        * **Thought:** I need recent articles about advancements in renewable energy.
        * **Action:** Use Tavily Web Search with the query "latest advancements in renewable energy technologies" and set `time_range` to "month".
        * **Observation:** Retrieved 10 articles discussing recent developments in solar panels, wind turbines, and energy storage.
        * **Thought:** To gain deeper insights, I'll crawl a relevant industry-leading renewable energy site.
        * **Action:** Use Tavily Web Crawl on the URL of a leading renewable energy industry website, setting `max_depth` to 2.
        * **Observation:** Gathered extensive content from multiple articles linked on the site, highlighting new technologies and innovations.
        * **Final Answer:** Provide a synthesized summary of findings with citations.

        ---

        You will now receive a research question from the user:

        """
        ),
        MessagesPlaceholder(variable_name="messages"),
        ]
    ),name="web_agent",
)

from langchain.schema import HumanMessage

# Test the agent with a sample question
inputs = {
    "messages": [
        HumanMessage(
            content="find all the iphone models currently available on apple.com and their prices"
        )
    ]
}

inputs = {
    "messages": [
        HumanMessage(
            content="find top 5 jobs listed for a software engineer in jobsnepal.com"
        )
    ]
}

for output in web_agent.stream(inputs, stream_mode="values"):
    messages = output["messages"][-1]
    if isinstance(messages, tuple):
        print(messages.content)
    else:
        messages.pretty_print()
    # print(output.content)