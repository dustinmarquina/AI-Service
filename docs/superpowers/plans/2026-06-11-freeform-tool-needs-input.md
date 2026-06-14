# Freeform Tool `needs_input` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend tool-step interrupt/resume so a tool can ask the user for one missing free-form field such as `description` or `amount`, not just a candidate selection.

**Architecture:** Keep the existing `tool_result["status"] == "needs_input"` contract, but add an explicit `input_kind` discriminator. `selection` keeps the current candidate-based flow; new `text` mode stores a paused step with a target field name, accepts the user's next reply as plain text, injects that value back into the saved args, and retries the same tool step. `create_transaction` becomes the first consumer by returning `needs_input` for missing `amount` or `description` instead of failing only through planner-level clarify.

**Tech Stack:** Python, LangGraph, FastMCP, unittest

---

### Task 1: Lock the contract in tests before changing runtime behavior

**Files:**
- Modify: `tests/test_executor.py`
- Modify: `tests/test_mcp_server_mongo.py`

- [ ] **Step 1: Add a failing executor test for free-form text resume**

Add a test near the existing tool pending-input tests that starts with a paused tool step:

```python
def test_tool_step_resumes_after_text_input(self):
    fake_registry = types.SimpleNamespace(
        call_mcp_tool=AsyncMock(return_value={"status": "success", "data": {"id": "txn-1"}})
    )
    sys.modules["agents.orchestrator.graph.tool_registry"] = fake_registry

    original_interrupt = EXECUTOR_MODULE.interrupt
    EXECUTOR_MODULE.interrupt = lambda _value: "bún chả"
    try:
        state = {
            "steps": [
                {"id": "s1", "type": "tool", "name": "create_transaction", "args": {"amount": "51k"}},
            ],
            "step_index": 0,
            "step_results": {
                "s1__pending_input": {
                    "input_kind": "text",
                    "field": "description",
                    "prompt_intro": "Khoản 51k này là cho gì?",
                    "args": {"amount": "51k", "user_id": "user-123"},
                }
            },
            "past_steps": [],
            "user_id": "user-123",
        }

        result = asyncio.run(EXECUTOR_MODULE.executor_node(state))
    finally:
        EXECUTOR_MODULE.interrupt = original_interrupt

    fake_registry.call_mcp_tool.assert_awaited_once_with(
        "create_transaction",
        {"amount": "51k", "user_id": "user-123", "description": "bún chả"},
    )
    assert result["step_index"] == 1
```

- [ ] **Step 2: Add a failing executor test for malformed text `needs_input` contract**

Add a second test that returns:

```python
{
  "status": "needs_input",
  "input_kind": "text",
  "prompt": "Khoản này là cho gì?"
}
```

and assert executor treats it as invalid because `field` is missing:

```python
self.assertEqual(result["past_steps"][0]["status"], "error")
self.assertIn("invalid format", result["messages"][0].content.lower())
```

- [ ] **Step 3: Add a failing MCP-server test for `create_transaction` missing description**

Add a test in `tests/test_mcp_server_mongo.py` that calls:

```python
result = asyncio.run(
    MCP_SERVER_MODULE.create_transaction(
        amount="51k",
        description="",
        user_id="user-123",
    )
)
```

and expect:

```python
self.assertEqual(result["status"], "needs_input")
self.assertEqual(result["input_kind"], "text")
self.assertEqual(result["field"], "description")
```

- [ ] **Step 4: Run the focused failing tests**

Run:

```bash
python -m unittest tests/test_executor.py tests/test_mcp_server_mongo.py
```

Expected: FAIL in the three new tests because executor and `create_transaction` do not support text-mode `needs_input` yet.

- [ ] **Step 5: Commit the red tests**

```bash
git add tests/test_executor.py tests/test_mcp_server_mongo.py
git commit -m "test: cover freeform tool needs_input flow"
```

### Task 2: Extend executor pending-input handling to support `input_kind="text"`

**Files:**
- Modify: `agents/orchestrator/graph/executor.py`
- Test: `tests/test_executor.py`

- [ ] **Step 1: Add a small helper to normalize text replies**

Add a helper near the selection helpers:

```python
def _resolve_text_input(reply: Any) -> str | None:
    if isinstance(reply, dict):
        for value in reply.values():
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None
    if isinstance(reply, str) and reply.strip():
        return reply.strip()
    return None
```

- [ ] **Step 2: Add a helper for generic free-form prompting**

Add an async helper next to `_prompt_for_selection(...)`:

```python
async def _prompt_for_text_input(prompt_intro: str) -> str | None:
    prompt = prompt_intro.strip() or "Please provide the missing information:"
    while True:
        reply = interrupt(prompt)
        if reply == prompt:
            return None
        text = _resolve_text_input(reply)
        if text:
            return text
        prompt = f"{prompt_intro}\nPlease type a short text answer."
```

- [ ] **Step 3: Handle paused text-input steps before the existing selection branch**

In the `if pending_input:` branch of `executor_node(...)`, branch first on `input_kind`:

```python
input_kind = str(pending_input.get("input_kind") or "selection").strip().lower()
if input_kind == "text":
    field = str(pending_input.get("field") or "").strip()
    prompt_intro = str(pending_input.get("prompt_intro") or pending_input.get("prompt") or "").strip()
    text_value = await _prompt_for_text_input(prompt_intro)
    if text_value is None:
        return {
            "step_index": step_index,
            "step_results": step_results,
            "past_steps": past_steps,
        }
    if not field:
        past_steps = _append_past_step(
            past_steps,
            step_id=step_id,
            step_type="tool",
            step_input=pending_input.get("args", {}),
            output={"error": "Tool requested free-form input without a target field"},
            status="error",
            summary="Tool requested input in an invalid format",
            reasoning=step_reasoning,
        )
        return {
            "messages": [AIMessage(content="The tool requested input in an invalid format.")],
            "step_index": len(steps),
            "step_results": step_results,
            "past_steps": past_steps,
        }

    resolved_args = dict(pending_input.get("args") or {})
    resolved_args[field] = text_value
    step_results.pop(f"{step_id}__pending_input", None)
```

- [ ] **Step 4: Mark selection-mode pending state explicitly**

When storing pending input for existing selection-based tool interrupts, add:

```python
"input_kind": "selection",
```

to the `step_results[f"{step_id}__pending_input"]` payload so resume code is explicit instead of inferred.

- [ ] **Step 5: Validate text-mode `needs_input` from fresh tool results**

In the branch:

```python
if isinstance(tool_result, dict) and tool_result.get("status") == "needs_input":
```

split on:

```python
input_kind = str(tool_result.get("input_kind") or "selection").strip().lower()
```

For `text` mode, require:
- `field`
- `prompt` or `prompt_intro`

and store:

```python
step_results[f"{step_id}__pending_input"] = {
    "input_kind": "text",
    "field": field,
    "prompt_intro": prompt_intro,
    "prompt": prompt_intro,
    "args": resolved_args,
}
```

Then call:

```python
text_value = await _prompt_for_text_input(prompt_intro)
```

If received, inject it into `resolved_args[field]` and retry the tool exactly once, mirroring the current selection-resume path.

- [ ] **Step 6: Run the focused executor tests**

Run:

```bash
python -m unittest tests/test_executor.py
```

Expected: PASS, including the new text-mode tests and all existing selection-mode tests.

- [ ] **Step 7: Commit executor support**

```bash
git add agents/orchestrator/graph/executor.py tests/test_executor.py
git commit -m "feat: add freeform tool needs_input resume"
```

### Task 3: Make `create_transaction` the first real text-input tool

**Files:**
- Modify: `agents/orchestrator/mcp_server.py`
- Test: `tests/test_mcp_server_mongo.py`

- [ ] **Step 1: Relax signature validation for fields you want the tool to request**

Update the `create_transaction` signature so the tool body can run even when a field is missing:

```python
async def create_transaction(
    amount: str = "",
    description: str = "",
    wallet_id: str = "",
    ...
) -> Dict[str, Any]:
```

This is necessary because if `description` stays required at schema level, Pydantic rejects the call before the tool can return `needs_input`.

- [ ] **Step 2: Replace early hard error on missing description with text-mode `needs_input`**

At the top of `create_transaction(...)`, change:

```python
if not str(description).strip():
    return _error_response("Description is required.")
```

to:

```python
if not str(description).strip():
    parsed_amount = _parse_amount(amount)
    amount_text = str(parsed_amount if parsed_amount is not None else amount).strip()
    return {
        "status": "needs_input",
        "input_kind": "text",
        "field": "description",
        "prompt": f"Khoản {amount_text} này là cho gì?",
    }
```

Keep `amount` as an immediate error for now unless you also want the tool to ask for amount in this same stage.

- [ ] **Step 3: Leave wallet behavior unchanged in this stage**

Do not remove wallet lookup or `request_text` yet. Stage 3 is only about free-form missing input. Keep the rest of `create_transaction` behavior unchanged so you isolate the change set.

- [ ] **Step 4: Add a second MCP-server test for missing amount only if you decide to support it now**

If you want amount prompting in this stage too, add:

```python
self.assertEqual(result["field"], "amount")
```

Otherwise skip this step and keep amount as a hard error.

- [ ] **Step 5: Run the MCP-server tests**

Run:

```bash
python -m unittest tests/test_mcp_server_mongo.py
```

Expected: PASS, with `create_transaction` now returning text-mode `needs_input` for missing description.

- [ ] **Step 6: Commit tool-side support**

```bash
git add agents/orchestrator/mcp_server.py tests/test_mcp_server_mongo.py
git commit -m "feat: let create_transaction request missing description"
```

### Task 4: Regression pass across the graph paths that touch paused tool steps

**Files:**
- Test: `tests/test_executor.py`
- Test: `tests/test_replanner.py`
- Test: `tests/test_main_graph_rewoo.py`

- [ ] **Step 1: Run the graph-adjacent suites**

Run:

```bash
python -m unittest tests/test_executor.py tests/test_replanner.py tests/test_main_graph_rewoo.py
```

Expected: PASS. Existing `selection`-mode interrupt behavior must remain unchanged.

- [ ] **Step 2: Run the broader targeted regression set**

Run:

```bash
python -m unittest tests/test_planner.py tests/test_tool_registry.py tests/test_mcp_server_mongo.py tests/test_graph_db.py
```

Expected: PASS. The free-form input change should not affect planner, tool arg coercion, or SQL execution.

- [ ] **Step 3: Manual verification scenario**

Run your local graph and verify this exact flow:

1. user: `51k`
2. agent: clarify or idle, depending on planner
3. user: `thêm giao dịch`

This should still **not** create a transaction.

Then verify:

1. user: `51k bún chả`
2. if wallet is unresolved, normal wallet-selection path still works

And verify the new path directly:

1. force a call into `create_transaction` with `amount="51k"` and empty `description`
2. tool returns `needs_input` text prompt
3. user replies `bún chả`
4. same tool step resumes and succeeds

- [ ] **Step 4: Commit regression confirmation**

```bash
git add tests/test_executor.py tests/test_mcp_server_mongo.py
git commit -m "test: verify freeform tool interrupt regression paths"
```

### Task 5: Document the contract boundary so later cleanup is easier

**Files:**
- Modify: `agents/orchestrator/graph/executor.py`
- Modify: `agents/orchestrator/mcp_server.py`

- [ ] **Step 1: Add a short comment above the tool `needs_input` handling branch**

Add a brief comment like:

```python
# Tool interrupts support two contracts:
# - selection: choose one candidate row and map it into a target arg
# - text: collect one free-form field value and retry the same tool step
```

- [ ] **Step 2: Add a short docstring note in `create_transaction(...)`**

Add one sentence explaining that missing `description` now returns text-mode `needs_input` so executor can resume the same tool step after one user reply.

- [ ] **Step 3: Commit the documentation cleanup**

```bash
git add agents/orchestrator/graph/executor.py agents/orchestrator/mcp_server.py
git commit -m "docs: clarify tool needs_input contracts"
```

---

**Notes for the next stage**

Do **not** try to solve these in the same implementation batch:
- planner-level `pending_clarification`
- removal of `request_text`
- removal of tool-side wallet resolution

Those are separate architectural fixes. This plan is only for making paused tool steps capable of collecting one missing free-form field safely.
