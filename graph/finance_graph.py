from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from mcp_service.mcp_client import create_finance_tool_node
from langchain_core.tools import tool
from langgraph.graph import START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class FinanceState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


async def build_finance_graph():
    tools, tool_node = await create_finance_tool_node()

    @tool
    def retrieve_financial_knowledge(query: str) -> str:
        """
        Retrieve relevant financial advisory knowledge based on the user query.
        This is a prototype RAG tool. Replace the mock logic with vector search later.
        """
        # TODO: Replace with vector database similarity search
        mock_knowledge_base = {
            "emergency fund": "An emergency fund should cover 3 to 6 months of essential expenses.",
            "savings rate": "A healthy savings rate is typically at least 20% of monthly income.",
            "debt ratio": "Debt-to-income ratio above 40% is considered financially risky."
        }

        for key, value in mock_knowledge_base.items():
            if key in query.lower():
                return value

        return "No relevant financial guideline found."

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.3,
        api_key="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",
    )

    # Add RAG retrieval tool to existing MCP tools
    enhanced_tools = tools + [retrieve_financial_knowledge]

    llm_with_tools = llm.bind_tools(enhanced_tools)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI financial advisory assistant.

When the user asks for financial advice, guidelines, or best practices:
- First call the `retrieve_financial_knowledge` tool to obtain authoritative guidance.
- Then combine retrieved knowledge with user-specific financial data (if available).
- Provide structured and explainable recommendations.
- Answers MUST be in Vietnamese.

Use tools whenever external data or structured computation is required.
"""),
        MessagesPlaceholder("messages"),
    ])

    chat_chain = prompt | llm_with_tools

    def chat_node(state: FinanceState):
        response = chat_chain.invoke({
            "messages": state["messages"]
        })
        return {"messages": [response]}

    graph = StateGraph(FinanceState)

    graph.add_node("chat", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat")
    graph.add_conditional_edges("chat", tools_condition)
    graph.add_edge("tools", "chat")

    return graph.compile()

# async def create_graph(session):
#     tools = await load_mcp_tools(session)
#     llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key="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")
#     llm_with_tools = llm.bind_tools(tools)

#     prompt_template = ChatPromptTemplate.from_messages([
#         ("system", "You are a helpful assistant that uses tools to explore Wikipedia."),
#         MessagesPlaceholder("messages")
#     ])

#     chat_llm = prompt_template | llm_with_tools

#     def chat_node(state: State) -> State:
#         state["messages"] = chat_llm.invoke({"messages": state["messages"]})
#         return state

#     graph = StateGraph(State)
#     graph.add_node("chat_node", chat_node)
#     graph.add_node("tool_node", ToolNode(tools=tools))
#     graph.add_edge(START, "chat_node")
#     graph.add_conditional_edges("chat_node", tools_condition, {
#         "tools": "tool_node",
#         "__end__": END
#     })
#     graph.add_edge("tool_node", "chat_node")

#     return graph.compile(checkpointer=MemorySaver())