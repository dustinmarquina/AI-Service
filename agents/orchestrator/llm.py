from langchain_groq import ChatGroq
import os

def get_classifier_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.0,
        api_key=os.getenv("groq_api_key"),
    )

def get_sql_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.0,
        api_key=os.getenv("groq_api_key"),
    )

def get_extractor_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.0,
        api_key=os.getenv("groq_api_key"),
    )

def get_generator_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.7,
        api_key=os.getenv("groq_api_key"),
    )


