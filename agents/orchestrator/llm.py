import json
import os
from typing import Any

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_groq import ChatGroq


def get_classifier_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("groq_api_key") or os.getenv("GROQ_API_KEY"),
        timeout=15.0,
        temperature=0.0,
    )


    

def get_sql_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("groq_api_key") or os.getenv("GROQ_API_KEY"),
        timeout=15.0,
        temperature=0.2,
    )


def get_extractor_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("groq_api_key") or os.getenv("GROQ_API_KEY"),
        timeout=15.0,
        temperature=0.2,
    )

def get_generator_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("groq_api_key") or os.getenv("GROQ_API_KEY"),
        timeout=15.0,
        temperature=0.7,
    )



# def get_9router_llm():
    # return NineRouterLLM()


