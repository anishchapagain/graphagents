# HITL: Human in the Loop with LangSmith and Groq
# This example demonstrates how to use LangSmith to create a human-in-the-loop

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import END, START
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv

import os

load_dotenv()

os.environ["LANGSMITH_TRACING"]="true"
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"]="langchain_agents"

class State(TypedDict): 
    messages: Annotated[list, add_messages]

from langchain.chat_models import init_chat_model
llm=init_chat_model("groq:llama3-8b-8192")
# llm=init_chat_model("groq:llama3-70b-8192")
# llm=init_chat_model("groq:qwen-7b-8192")
# llm=init_chat_model("groq:comppound-beta")

# TODO