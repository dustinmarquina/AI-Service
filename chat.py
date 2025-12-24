
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
    """
    Generate a reply to the user's message based on chat history.
    """
    chat_history = get_chat_history(user_id)
    
    # Update chat history with user's message
    chat_history = update_chat_history(chat_history, "user", user_message)
    
    # Here you would integrate with your LLM to generate a response
    # For demonstration, we'll use a placeholder response
    assistant_response = f"Echo: {user_message}"
    
    # Update chat history with assistant's response
    chat_history = update_chat_history(chat_history, "assistant", assistant_response)
    
    # Save updated chat history
    set_chat_history(user_id, chat_history)
    
    return assistant_response