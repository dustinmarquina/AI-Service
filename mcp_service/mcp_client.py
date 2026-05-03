from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.prebuilt import ToolNode


async def create_finance_tool_node():
    """
    Creates a LangGraph ToolNode connected to the Finance MCP server.
    This node can be added directly into a LangGraph workflow.
    Returns tuple of (tools, tool_node)
    """
    client = MultiServerMCPClient(
        {
            "Finance-ai": {
                "transport": "stdio",
                "command": "python",
                "args": ["-m", "mcp_service.server"],
            }
        }
    )

    tools = await client.get_tools()

    return tools, ToolNode(tools)


async def create_finance_agent():
    """
    Optional: If you still want an agent layer (LLM + tools).
    """
    client = MultiServerMCPClient(
        {
            "Finance-ai": {
                "transport": "stdio",
                "command": "python",
                "args": ["-m", "mcp_service.server"],
            }
        }
    )

    tools = await client.get_tools()

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.3,
        api_key="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",
    )

    return create_agent(llm, tools)

# 