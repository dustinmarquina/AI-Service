import os
from langgraph.graph import StateGraph, END
from langchain_community.utilities import SQLDatabase
from .state import State
from ..llm import get_sql_llm

db = SQLDatabase.from_uri(os.getenv("POSTGRES_URI"))


# --- 1. Generate SQL ---
async def generate_sql_node(state: State):
    messages = state.get("messages", [])

    if not messages:
        return {"response": "Không nhận được input từ người dùng."}

    user_input = messages[-1].content

    llm = get_sql_llm()

    # Include previous SQL error in the prompt if exists, to help LLM avoid repeating the same mistake
    previous_error = state.get("sql_error")
    error_context = f"Previous SQL error: {previous_error}" if previous_error else ""
    prompt = f"""
You are an agent designed to interact with a PostgreSQL database.

Schema:
users(id, email, first_name, last_name, created_at, updated_at)

Instructions:
- Given a user question, generate a syntactically correct PostgreSQL SELECT query.
- Only query relevant columns, NEVER use SELECT *.
- Limit results to at most 5 rows unless counting.
- If the question implies counting (e.g., "bao nhiêu"), use COUNT(*).
- Do NOT use any INSERT, UPDATE, DELETE, DROP.
- Ensure column names exist in the schema.
- Return ONLY the SQL query, no explanation.

User question:
"{user_input}"
"""

    response = await llm.ainvoke(prompt)

    print("LLM RAW RESPONSE:", response)

    # handle both AIMessage and raw string cases
    if hasattr(response, "content"):
        sql = (response.content or "").strip()
    else:
        sql = str(response).strip()

    print("LLM CONTENT:", sql)

    # Reset stale values from previous turns so format node does not reuse old response.
    return {
        "sql_query": sql,
        "sql_results": None,
        "sql_error": None,
        "response": None,
    }


# --- 2. Execute SQL ---
async def execute_sql_node(state: State):
    sql = state.get("sql_query")
    print("EXECUTE NODE - SQL to execute:", sql)

    #check if sql is empty or None
    if sql is None or str(sql).strip() == "":
        print("SQL is empty or None:", repr(sql))
        return {
            "response": "Không tạo được câu truy vấn SQL."
        }

    cleaned_sql = sql.strip()

    # handle cases like ```sql ... ```
    if cleaned_sql.startswith("```"):
        cleaned_sql = cleaned_sql.replace("```sql", "").replace("```", "").strip()

    if not cleaned_sql.lower().startswith("select"):
        return {
            "response": "Chỉ cho phép truy vấn SELECT."
        }

    sql = cleaned_sql

    try:
        print("EXECUTING SQL:", sql)
        result = db.run(sql)
        print("RAW RESULT:", result)
        return {
            "sql_results": result,
            "response": None,
        }
    except Exception as e:
        return {
            "sql_results": None,
            "sql_error": str(e),
            "response": f"Có lỗi khi thực thi SQL: {str(e)}"
        }


# --- 3. Format response ---
async def format_response_node(state: State):
    # if previous node already returned response → keep it
    if state.get("response"):
        print("FORMAT NODE - response:", state.get("response"))
        return {"response": state["response"]}

    results = state.get("sql_results")
    print("FORMAT NODE - response:", state.get("response"))
    print("FORMAT NODE - results:", results)

    if not results:
        return {"response": "Không có dữ liệu."}

    # call LLM if results length is greater than 1 or if the result is not a simple count
    if len(results) > 1 or (len(results) == 1 and not isinstance(results[0][0], int)):
        llm = get_sql_llm()
        prompt = f"""Bạn là một trợ lý giúp giải thích kết quả truy vấn SQL cho người dùng.
Dữ liệu trả về từ truy vấn SQL là:
{results}
Chỉ cần trả lời một câu ngắn gọn giải thích ý nghĩa của dữ liệu này, tránh thuật ngữ kỹ thuật. Trả lời bằng tiếng Việt.
"""

        response = await llm.ainvoke(prompt)
        return {"response": response}


    return {
        "response": f"Kết quả: {results}"
    }

def route_by_sql_error(state: State):
    if state.get("sql_error"):
        return "generate_sql"
    else:
        return "execute_sql"

# --- Build SQL Graph ---
def build_sql_graph():
    graph = StateGraph(State)

    graph.add_node("generate_sql", generate_sql_node)
    graph.add_node("execute_sql", execute_sql_node)
    graph.add_node("format_response", format_response_node)

    graph.set_entry_point("generate_sql")

    # Conditional routing: if sql_error is not None -> recursively call generate_sql, else go to execute_sql
    graph.add_conditional_edges(
        "generate_sql",
        route_by_sql_error,
        {
            "generate_sql": "generate_sql",
            "execute_sql": "execute_sql"
        }
    )
    graph.add_edge("execute_sql", "format_response")
    graph.add_edge("format_response", END)

    return graph.compile()
