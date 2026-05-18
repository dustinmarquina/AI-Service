import json
import os
from typing import Any

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_groq import ChatGroq


# class NineRouterLLM:
#     def __init__(
#         self,
#         model: str = "groq/openai/gpt-oss-120b",
#         timeout: float = 60.0,
#         temperature: float = 0.2,
#         tools: list[Any] | None = None,
#     ):
#         base = os.getenv("9router_api_base_url") or os.getenv("9ROUTER_API_BASE_URL")
#         if not base:
#             base = "https://9router.manportfolio.id.vn"

#         host = base.rstrip("/")
#         if host.endswith("/v1"):
#             host = host[:-3].rstrip("/")

#         self.base_url = host
#         self.url = host + "/v1/chat/completions"
#         self.api_key = (
#             os.getenv("9router_api_key")
#             or os.getenv("9ROUTER_API_KEY")
#             or os.getenv("groq_api_key")
#             or os.getenv("GROQ_API_KEY")
#         )
#         self.model = model
#         self.model_name = model
#         self.timeout = timeout
#         self.temperature = temperature
#         self._tools = list(tools or [])

#     def bind_tools(self, tools: list[Any]) -> "NineRouterLLM":
#         return NineRouterLLM(
#             model=self.model,
#             timeout=self.timeout,
#             temperature=self.temperature,
#             tools=tools,
#         )

#     def _headers(self) -> dict[str, str]:
#         headers = {"Content-Type": "application/json"}
#         if self.api_key:
#             headers["Authorization"] = f"Bearer {self.api_key}"
#         return headers

#     def _to_text(self, value: Any) -> str:
#         if value is None:
#             return ""
#         if isinstance(value, str):
#             return value
#         return json.dumps(value, ensure_ascii=False)

#     def _message_to_openai(self, message: Any) -> dict[str, Any]:
#         if isinstance(message, dict):
#             role = str(message.get("role", "user"))
#             content = self._to_text(message.get("content", ""))
#             out: dict[str, Any] = {"role": role, "content": content}
#             if role == "tool" and message.get("tool_call_id"):
#                 out["tool_call_id"] = message.get("tool_call_id")
#             return out

#         if isinstance(message, SystemMessage):
#             return {"role": "system", "content": self._to_text(message.content)}
#         if isinstance(message, HumanMessage):
#             return {"role": "user", "content": self._to_text(message.content)}
#         if isinstance(message, ToolMessage):
#             return {
#                 "role": "tool",
#                 "content": self._to_text(message.content),
#                 "tool_call_id": message.tool_call_id,
#             }
#         if isinstance(message, AIMessage):
#             out = {"role": "assistant", "content": self._to_text(message.content)}
#             if getattr(message, "tool_calls", None):
#                 raw_tool_calls = []
#                 for call in message.tool_calls:
#                     raw_tool_calls.append(
#                         {
#                             "id": call.get("id"),
#                             "type": "function",
#                             "function": {
#                                 "name": call.get("name", ""),
#                                 "arguments": json.dumps(call.get("args", {}), ensure_ascii=False),
#                             },
#                         }
#                     )
#                 out["tool_calls"] = raw_tool_calls
#             return out
#         if isinstance(message, BaseMessage):
#             role = "assistant" if message.type == "ai" else "user"
#             return {"role": role, "content": self._to_text(message.content)}

#         return {"role": "user", "content": self._to_text(message)}

#     def _convert_tools(self) -> list[dict[str, Any]]:
#         converted: list[dict[str, Any]] = []
#         for tool in self._tools:
#             try:
#                 converted.append(convert_to_openai_tool(tool))
#             except Exception:
#                 name = getattr(tool, "name", "tool")
#                 description = getattr(tool, "description", "")
#                 converted.append(
#                     {
#                         "type": "function",
#                         "function": {
#                             "name": name,
#                             "description": description,
#                             "parameters": {"type": "object", "properties": {}},
#                         },
#                     }
#                 )
#         return converted

#     def _payload(self, input_data: Any, stream: bool) -> dict[str, Any]:
#         if isinstance(input_data, str):
#             messages = [{"role": "user", "content": input_data}]
#         elif isinstance(input_data, list):
#             messages = [self._message_to_openai(m) for m in input_data]
#         else:
#             messages = [{"role": "user", "content": self._to_text(input_data)}]

#         payload: dict[str, Any] = {
#             "model": self.model,
#             "messages": messages,
#             "temperature": self.temperature,
#             "stream": stream,
#         }
#         if self._tools:
#             payload["tools"] = self._convert_tools()
#             payload["tool_choice"] = "auto"
#         return payload

#     def _to_ai_message(self, data: dict[str, Any]) -> AIMessage:
#         message = data.get("choices", [{}])[0].get("message", {})
#         content = message.get("content") or ""
#         tool_calls = []

#         for tool_call in message.get("tool_calls", []) or []:
#             fn = tool_call.get("function", {})
#             raw_args = fn.get("arguments") or "{}"
#             try:
#                 args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
#             except json.JSONDecodeError:
#                 args = {}
#             tool_calls.append(
#                 {
#                     "name": fn.get("name", ""),
#                     "args": args,
#                     "id": tool_call.get("id") or "",
#                     "type": "tool_call",
#                 }
#             )

#         return AIMessage(content=content, tool_calls=tool_calls)

#     def invoke(self, input_data: Any) -> AIMessage:
#         payload = self._payload(input_data, stream=False)
#         with httpx.Client(timeout=self.timeout) as client:
#             resp = client.post(self.url, json=payload, headers=self._headers())
#             resp.raise_for_status()
#             return self._to_ai_message(resp.json())

#     async def ainvoke(self, input_data: Any) -> AIMessage:
#         payload = self._payload(input_data, stream=False)
#         async with httpx.AsyncClient(timeout=self.timeout) as client:
#             resp = await client.post(self.url, json=payload, headers=self._headers())
#             resp.raise_for_status()
#             return self._to_ai_message(resp.json())

#     def stream(self, prompt: str):
#         payload = self._payload(prompt, stream=True)
#         with httpx.Client(timeout=self.timeout) as client:
#             with client.stream("POST", self.url, json=payload, headers=self._headers()) as resp:
#                 resp.raise_for_status()
#                 for chunk in resp.iter_text(chunk_size=1024):
#                     if chunk:
#                         yield chunk

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


