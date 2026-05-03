from predictor import prepare_analysis_context, prepare_budget_context, prepare_prediction_context 

OLLAMA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_name",
            "description": "Lấy tên người dùng",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_age",
            "description": "Lấy tuổi người dùng",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]

def get_name():
    return "Gia Thinh"

def get_age():
    return 21

available_tools = {
    "get_name": get_name,
    "get_age": get_age,
}

def chat_reply(messages):
    from ollama import chat

    final_text = ""

    res = chat(
        model="llama3.1",
        messages=messages,
        tools=OLLAMA_TOOLS,
        stream=True
    )

    for part in res:
        msg = part.get("message", {})

        if "tool_calls" in msg:
            for call in msg["tool_calls"]:
                tool_name = call["function"]["name"]
                args = call["function"].get("arguments", {}) or {}

                result = available_tools[tool_name](**args)

                messages.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": str(result)
                })

            return chat_reply(messages)  # recurse safely

        else:
            final_text += msg.get("content", "")

    return final_text
    
MAX_TURN = 6

CHAT_MEMORY = {}

def update_chat_history(chat_history: list, role: str, content: str) -> list:
    """
    Update the chat history with a new message.
    Ensures the chat history does not exceed MAX_TURN exchanges.
    """
    chat_history.append({"role": role, "content": content})
    
    # Each turn consists of 2 messages (user and assistant)
    if len(chat_history) > MAX_TURN * 2:
        # Remove the oldest turn (first two messages)
        chat_history = chat_history[-MAX_TURN * 2:]
    
    return chat_history

def get_chat_history(user_id: str) -> list:
    """
    Retrieve the chat history for a given user.
    Initializes an empty history if none exists.
    """
    if user_id not in CHAT_MEMORY:
        CHAT_MEMORY[user_id] = []
    return CHAT_MEMORY[user_id]

def set_chat_history(user_id: str, chat_history: list):
    """
    Set the chat history for a given user.
    """
    CHAT_MEMORY[user_id] = chat_history

def reply_to_user(user_id: str, user_message: str) -> str:
    from llm_client import call_local_llm
    """
    Generate a reply to the user's message based on chat history.
    """
    chat_history = get_chat_history(user_id)
    
    # Update chat history with user's message
    chat_history = update_chat_history(chat_history, "user", user_message)

    with open("prompts/system.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()

    messages = [
        {"role": "system", "content": system_prompt},
        *chat_history
    ]
    
   
    def generate_response():
        buffer = ""
        for chunk in call_local_llm(
            prompt="\n".join([m["content"] for m in messages]),
            temperature=0.5,
            stream=True
        ):
            if chunk:
                buffer += chunk
                if len(buffer) >= 50 and chunk in [' ', '\n', '.', '!', '?', '|', ')']:
                    yield buffer
                    buffer = ""
        if buffer:
            yield buffer
    # Update chat history with assistant's response
    chat_history = update_chat_history(messages, "assistant", "".join(generate_response()))
    
    # Save updated chat history
    set_chat_history(user_id, chat_history)
    
    return "".join(generate_response())