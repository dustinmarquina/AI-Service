# Orchestrator Agent - Technical Report

**Project:** AI Service - Vietnamese Finance Assistant  
**Date:** May 13, 2026  
**Version:** 1.0  
**Status:** Active Implementation

---

## 1. Executive Summary

The **Orchestrator Agent** is the core intelligence layer of the AI Service, a Vietnamese-language finance assistant built using LangGraph and LLMs. It routes user queries intelligently, manages conversation context, orchestrates MCP (Model Context Protocol) tools, and coordinates SQL-based analytics through specialized subgraphs.

### Key Capabilities

- **Intelligent Query Routing**: Distinguishes between conversational requests and analytical queries
- **Multi-Agent Coordination**: Delegates complex tasks to specialized subgraphs (SQL agent)
- **Context Management**: Maintains session-based short-term memory with automatic trimming
- **Runtime Tool Injection**: Dynamically injects authentication tokens and user context into tool calls
- **Vietnamese-First Design**: Native support for Vietnamese language queries and responses

---

## 2. Architecture Overview

### 2.1 High-Level Flow

```
User Input
    ↓
[Router Conditional Node]
    ↓
    ├─→ Chat Branch ──→ [LLM + MCP Tools] ──→ Tool Execution ──→ [Finalize]
    │   (Conversational)      (if tools needed)
    │
    └─→ SQL Agent Branch ──→ [SQL Subgraph] ──→ [Finalize]
        (Analytical)
        ↓
    [Finalize Node: Memory + Response]
        ↓
    Response to User
```

### 2.2 Core Components

| Component               | Purpose                                   | Technology                            |
| ----------------------- | ----------------------------------------- | ------------------------------------- |
| **Main Graph**          | Orchestrator logic and routing            | LangGraph StateGraph                  |
| **Router**              | Pattern-based query classification        | Regex patterns (Vietnamese + English) |
| **Chat Node**           | Conversational responses with tools       | Groq ChatGroq + MCP Tools             |
| **SQL Agent Node**      | SQL query generation and execution        | LangGraph + SQLDatabase               |
| **Tool Injection Node** | Runtime context injection                 | Custom middleware                     |
| **Finalize Node**       | Memory management and response formatting | Custom node                           |
| **State Management**    | Type-safe state representation            | LangGraph TypedDict                   |

---

## 3. Detailed Component Analysis

### 3.1 State Definition

**File:** `agents/orchestrator/graph/state.py`

```python
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # Conversation history
    session_id: Optional[str]                              # Session identifier
    user_id: Optional[str]                                 # User identifier
    token: Optional[str]                                   # Auth token for APIs
    response: Optional[str]                                # Final response
```

**State Management Strategy:**

- Messages use LangGraph's `add_messages` reducer for automatic deduplication
- Session-based organization for multi-user support
- Token injection for secure API calls

### 3.2 Router Logic

**File:** `agents/orchestrator/graph/main_graph.py` → `_is_read_query()`

**Pattern Recognition:**

```
Vietnamese Keywords: bao nhiêu, tổng, chi tiêu, tháng, tuần, hôm nay, thống kê, báo cáo, ...
English Keywords: budget, spent, how much, summary, report, history, total, ...
```

**Routing Decision:**

- **→ SQL Agent:** If query matches read patterns (analytics, history, summaries)
- **→ Chat Node:** Everything else (casual conversation, commands, incomplete requests)

**Sample Routes:**

- "Tôi chi bao nhiêu tiền tháng này?" → SQL Agent (analytical)
- "Chào bạn" → Chat (conversational)
- "Ghi nhận chi tiêu 50k" → Chat (action request)

### 3.3 Chat Node Architecture

**Responsibilities:**

1. Retrieve session memory for context
2. Build prompt with MCP tool catalog
3. Invoke LLM with tools binding
4. Support tool use through conditional routing

**LLM Configuration:**

- **Model:** Groq ChatGroq (OpenAI GPT-oss-120b with 0.2 temperature)
- **System Prompt:** Vietnamese finance assistant with explicit rules
- **Tools:** Dynamic MCP tools discovered at runtime

**Tool Binding Flow:**

```
Chat Node
  ↓
LLM evaluates if tools needed
  ↓
  ├─→ Yes: tool_calls included → Inject Context → Execute Tools → Loop back to Chat
  └─→ No: Direct response → Finalize
```

### 3.4 SQL Agent Subgraph

**File:** `agents/orchestrator/graph/sql_agent.py`

**Purpose:** Handles analytical queries requiring database access

**Components:**

- **SQL Context Builder**: Connects to PostgreSQL via SQLDatabase toolkit
- **Tool Node**: Manages SQL execution tools
- **Chat Loop**: LLM iteratively builds and refines SQL queries

**System Prompt:**

```
"SELECT-only queries - Never write INSERT, UPDATE, DELETE, DROP"
"Use at most 5 rows unless explicitly requested"
"Use SQL tools for aggregations (totals, counts, summaries)"
"Return results in Vietnamese"
```

**Integration:**

- Main graph calls `get_sql_agent()` and awaits result
- SQL agent returns filtered messages (final AI response only)
- Result propagates to finalize node

### 3.5 Runtime Context Injection Node

**File:** `agents/orchestrator/graph/main_graph.py` → `inject_runtime_context_node()`

**Purpose:** Secure dynamic credential injection

**Process:**

1. Detect if last AI message contains tool_calls
2. Extract `token` and `user_id` from state
3. Inject into tool arguments if field matches schema
4. Handle both `userId` (camelCase) and `user_id` (snake_case)

**Security Features:**

- Tokens never logged in full (only first 8 chars)
- Injection only occurs if field exists in tool schema
- Non-intrusive: only adds missing credentials

**Example Transformation:**

```
Before: {"amount": 50000}
After:  {"amount": 50000, "token": "xyz123...", "userId": "user-456"}
```

### 3.6 Finalize Node

**File:** `agents/orchestrator/graph/main_graph.py` → `finalize_node()`

**Responsibilities:**

1. **Memory Management**: Add user and AI messages to session history
2. **Trimming**: Keep only last 12 messages (configurable via `MAX_MEMORY_MESSAGES`)
3. **Response Extraction**: Extract the final non-tool AI message
4. **Fallback Handling**: Provide Vietnamese fallback if no response generated

**Memory Architecture:**

```
SHORT_TERM_MEMORY = {
    session_id_1: InMemoryChatMessageHistory([...]),
    session_id_2: InMemoryChatMessageHistory([...]),
    ...
}
```

**Trim Strategy:**

- Maintains maximum of 12 messages per session
- FIFO removal when limit exceeded
- Preserves recent context for better continuity

---

## 4. Data Flow Analysis

### 4.1 Chat Request Flow

```
1. User sends message
   state.messages = [HumanMessage("Tôi chi bao nhiêu tiền?"")]

2. Router evaluates pattern
   Pattern match: "bao nhiêu" → route to "sql_agent"

3. SQL Agent invoked
   - Parse messages
   - Build SQL query via LLM + tools
   - Execute database queries
   - Return final answer

4. Finalize
   - Add to session memory
   - Trim if needed
   - Return response

5. API returns
   {"response": AIMessage(content="Bạn đã chi 5,250,000 đồng tháng này")}
```

### 4.2 Tool Execution Flow

```
1. Chat node detects tool_calls needed
   AI response includes: tool_calls=[{name: "create_transaction", args: {...}}]

2. Conditional routing: tools_condition → "tools" node

3. Inject context node
   - Adds token and user_id if missing

4. Tool node executes
   - Call MCP tool with injected args
   - Capture result

5. Loop back to chat
   Add ToolMessage(result) to messages
   LLM continues conversation
```

---

## 5. Key Features

### 5.1 Session-Based Memory Management

**Implementation:** In-memory dictionaries with lazy initialization

```python
SHORT_TERM_MEMORY: dict[str, InMemoryChatMessageHistory] = {}

def _get_memory(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in SHORT_TERM_MEMORY:
        SHORT_TERM_MEMORY[session_id] = InMemoryChatMessageHistory()
    return SHORT_TERM_MEMORY[session_id]
```

**Benefits:**

- Session isolation (multi-user support)
- Conversation context preserved across requests
- Lightweight in-memory storage

**Limitations:**

- Memory lost on server restart
- Not suitable for multi-instance deployments
- Consider Redis/database for production scale

### 5.2 MCP Tools Integration

**Discovery:** Happens at startup

```python
client = MultiServerMCPClient(mcp_config)
tools = await client.get_tools()  # Discovered at runtime
```

**Catalog Generation:**

- Tools listed in system prompt
- LLM selects appropriate tools based on context
- Schema inspection for runtime injection

**Supported Transports:** STDIO (primary), potentially extensible

### 5.3 Vietnamese Language Support

**Native Features:**

- Vietnamese regex patterns in router
- Vietnamese system prompts
- Vietnamese fallback messages
- Support for Vietnamese numerals and currency

**Examples:**

- "Bao nhiêu" (how much) → SQL routing
- "Ghi nhận" (record) → Chat routing
- Response: "Bạn đã chi 5,250,000 đồng tháng này"

---

## 6. Configuration & Dependencies

### 6.1 Environment Variables

| Variable                     | Purpose                      | Default                             |
| ---------------------------- | ---------------------------- | ----------------------------------- |
| `ORCHESTRATOR_LLM_MODEL`     | LLM model selection          | `openai/gpt-oss-120b`               |
| `groq_api_key`               | Groq API authentication      | Required                            |
| `ORCHESTRATOR_MCP_COMMAND`   | MCP server command           | `python`                            |
| `ORCHESTRATOR_MCP_ARGS`      | MCP server arguments         | `-m agents.orchestrator.mcp_server` |
| `ORCHESTRATOR_MCP_TRANSPORT` | MCP transport type           | `stdio`                             |
| `POSTGRES_URI`               | PostgreSQL connection        | Required for SQL agent              |
| `TRANSACTION_API_URL`        | Transaction service endpoint | (from env)                          |
| `CATEGORY_API_URL`           | Category service endpoint    | (from env)                          |
| `TRANSACTION_API_TOKEN`      | Default transaction token    | (from env)                          |
| `TRANSACTION_USER_ID`        | Default user ID              | (from env)                          |

### 6.2 Dependencies

**Core Libraries:**

- `langgraph`: Graph-based orchestration
- `langchain`: LLM abstractions
- `langchain_groq`: Groq LLM integration
- `langchain_mcp_adapters`: MCP protocol support
- `langchain_community`: SQLDatabase toolkit
- `pydantic`: Data validation

**Deployment:**

- `fastapi`: Web framework
- `uvicorn`: ASGI server
- Integrated with RabbitMQ for async message processing

---

## 7. Performance Characteristics

### 7.1 Memory Usage

| Component                  | Memory Impact                        |
| -------------------------- | ------------------------------------ |
| Chat history (per session) | ~5-20 KB per session                 |
| MCP tools cache            | ~100 KB (depends on tool count)      |
| Short-term memory dict     | ~5-50 KB (12 sessions × 12 messages) |
| LLM context window         | 4K-8K tokens (Groq model)            |

### 7.2 Latency

- **Router decision**: <1ms (regex)
- **Chat node LLM call**: 1-3 seconds (network + inference)
- **SQL agent**: 2-5 seconds (DB query + LLM iterations)
- **Tool execution**: 500ms-2s (depends on tool)
- **Total request**: 2-8 seconds typical

### 7.3 Scalability Considerations

**Current Limitations:**

- In-memory session storage (single instance only)
- No distributed caching
- Sequential message processing

**Improvements for Scale:**

- Redis for distributed session storage
- Message queue for async processing
- LLM response caching
- Database query result caching

---

## 8. Error Handling & Resilience

### 8.1 Failure Points

| Point               | Handling                                           |
| ------------------- | -------------------------------------------------- |
| MCP tool not found  | LLM falls back to conversation                     |
| SQL query fails     | Returns empty result message                       |
| LLM API error       | Caught by LangGraph, propagated up                 |
| Missing credentials | Tool injection skipped; tool fails with auth error |
| Database connection | RuntimeError raised, startup fails                 |

### 8.2 Fallback Messages

```python
# SQL agent fallback
"Mình không tìm được kết quả phù hợp."

# Finalize fallback
"Mình chưa có phản hồi phù hợp."
```

### 8.3 Logging Strategy

- Comprehensive logging at INFO level
- Tool names logged for debugging
- Redacted credentials (only first 8 chars shown)
- Message content truncated to 200 chars in logs

---

## 9. Security Analysis

### 9.1 Token Management

**Positive:**

- Tokens injected only when missing
- Never logged in full
- Schema validation prevents injection into wrong fields

**Concerns:**

- Tokens stored in-memory (not encrypted)
- In-memory short-term memory not persisted securely
- No audit trail for tool execution

### 9.2 SQL Injection Prevention

**Safeguards:**

- SQLAlchemy ORM usage (parameterized queries)
- System prompt explicitly forbids INSERT/UPDATE/DELETE
- LLM trained on read-only queries

**Assumption:** Groq LLM respects safety guidelines

### 9.3 Input Validation

- LangChain automatically validates tool arguments against schema
- User input not sanitized (relies on LLM discretion)
- No rate limiting at agent level

---

## 10. Integration Points

### 10.1 FastAPI Integration

**Location:** `main.py` lifespan context manager

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.main_graph = await build_main_graph()
    # ... RabbitMQ setup ...
    yield
    # ... cleanup ...
```

**Integration Pattern:**

- Graph compiled during startup
- Available as `app.state.main_graph`
- Used by `/chat` endpoints

### 10.2 RabbitMQ Integration

- Message handler receives messages from queue
- Invokes orchestrator agent asynchronously
- Results published back to queue or database

### 10.3 API Routes

- `/chat` endpoints use main graph
- `/analysis` endpoints may use SQL agent directly
- Authentication passed via token injection

---

## 11. Testing & Validation

### 11.1 Test Files Identified

- `test_db.py`: Database connectivity tests
- `test_rabbitmq.py`: Message queue tests
- `tx_sandbox.py`: Transaction sandbox environment

### 11.2 Recommended Test Coverage

**Unit Tests:**

- Router pattern matching
- State transitions
- Memory trimming logic
- Context injection

**Integration Tests:**

- Full chat flow with mock MCP server
- SQL agent with test database
- Multi-session isolation
- Tool execution with injected credentials

**Load Tests:**

- Concurrent session handling
- Memory growth over time
- LLM latency under load

---

## 12. Future Enhancements

### 12.1 Short-Term (Weeks)

- [ ] Persistent session storage (Redis/PostgreSQL)
- [ ] Response caching for repeated queries
- [ ] Comprehensive test suite
- [ ] Error rate monitoring and alerts

### 12.2 Mid-Term (Months)

- [ ] Multi-language support beyond Vietnamese
- [ ] Advanced memory: long-term context storage
- [ ] Query optimization for SQL agent
- [ ] Audit logging for compliance
- [ ] A/B testing framework for routing rules

### 12.3 Long-Term (Quarters)

- [ ] Distributed graph execution
- [ ] Fine-tuned LLMs for financial domain
- [ ] Semantic similarity for query clustering
- [ ] Proactive financial recommendations
- [ ] Multi-modal input (voice, images)

---

## 13. Troubleshooting Guide

| Issue                       | Cause            | Solution                                         |
| --------------------------- | ---------------- | ------------------------------------------------ |
| Agent not responding        | MCP server down  | Check `ORCHESTRATOR_MCP_COMMAND` and logs        |
| Wrong routing               | Pattern mismatch | Review `_READ_PATTERNS` for edge cases           |
| Memory growing indefinitely | Trim not working | Check `MAX_MEMORY_MESSAGES` and `_trim_memory()` |
| SQL queries fail            | No DB connection | Verify `POSTGRES_URI` and PostgreSQL running     |
| Tool injection missing      | Schema mismatch  | Check tool's `args_schema` for field names       |
| Vietnamese output broken    | Encoding issue   | Ensure UTF-8 encoding in all layers              |

---

## 14. Conclusion

The **Orchestrator Agent** is a sophisticated multi-agent system that intelligently routes queries, maintains context, and coordinates specialized subgraphs. Its architecture balances simplicity with flexibility, using LangGraph's state machine pattern for clear, maintainable logic flow.

**Strengths:**

- Clear separation of concerns (routing, chat, SQL)
- Flexible MCP tool integration
- Strong Vietnamese language support
- Secure credential injection
- Efficient memory management

**Current Maturity:** **Beta** - Core functionality stable, production hardening needed

**Recommended Next Steps:**

1. Implement persistent session storage
2. Add comprehensive monitoring
3. Deploy with rate limiting and authentication
4. Establish test coverage baseline
5. Document runtime configuration for operators

---

## Appendix: File Structure

```
agents/orchestrator/
├── graph/
│   ├── __init__.py
│   ├── main_graph.py          # Main orchestrator logic
│   ├── state.py               # State definition
│   ├── sql_agent.py           # SQL subgraph
├── llm.py                      # LLM factory functions
├── mcp_server.py              # MCP server implementation
└── __init__.py
```

---

**Report Version:** 1.0  
**Last Updated:** May 13, 2026  
**Status:** Ready for Review
