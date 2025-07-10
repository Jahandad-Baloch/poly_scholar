from typing import Literal
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.types import Command
from langgraph.graph import MessagesState, StateGraph, START, END

# from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

def init_agent(params: dict):
    """
    Initialize a multi-agent system with a specified model and temperature.
    """
    model = params.get("model", "gemini-2.5-flash-lite-preview-06-17")
    temperature = params.get("temperature", 0.8)
    
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature
    )
