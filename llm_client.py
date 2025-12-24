import requests
from ollama import Client
import os

# Cloud Ollama configuration
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
OLLAMA_CLOUD_HOST = "https://ollama.com"
OLLAMA_LOCAL_HOST = os.getenv("OLLAMA_LOCAL_HOST", "http://localhost:11434")
# OLLAMA_LOCAL_FORWARD_HOST = "https://7m4plr3m-11434.asse.devtunnels.ms"

# -----------------------------------------------------
# 1. Low-level Ollama call
# -----------------------------------------------------

def call_local_llm(prompt: str, temperature: float = 0.2, stream: bool = False):
    """
    Call LLaMA via Ollama (cloud or local).
    Returns raw text output if stream=False, or yields chunks if stream=True.
    """
    try:
        if OLLAMA_API_KEY:
            # Use cloud Ollama with chat API
            client = Client(
                host=OLLAMA_CLOUD_HOST,
                headers={'Authorization': f'Bearer {OLLAMA_API_KEY}'}
            )
            
            # Cloud Ollama model (gpt-oss:120b or other available models)
            model_name = os.getenv("OLLAMA_MODEL", "gpt-oss:20b-cloud")
            
            # Convert prompt to chat message format
            messages = [{'role': 'user', 'content': prompt}]
            
            if stream:
                # Stream from cloud Ollama using chat API
                for part in client.chat(model_name, messages=messages, stream=True):
                    response_text = part.get('message', {}).get('content', '')
                    if response_text:
                        yield response_text
            else:
                # Non-streaming from cloud Ollama
                response = client.chat(model_name, messages=messages, stream=False)
                return response.get('message', {}).get('content', '')
        else:
            # Use local Ollama with generate API
            res = requests.post(
                f"{OLLAMA_LOCAL_HOST}/api/generate",
                json={
                    "model": "llama3.1",
                    "prompt": prompt,
                    "temperature": temperature,
                    "stream": stream
                },
                stream=stream
            )
            
            if res.status_code == 200:
                if stream:
                    # Yield chunks for streaming
                    for line in res.iter_lines():
                        if line:
                            try:
                                chunk_data = line.decode('utf-8')
                                import json
                                chunk_json = json.loads(chunk_data)
                                response_text = chunk_json.get("response", "")
                                if response_text:
                                    yield response_text
                            except json.JSONDecodeError as e:
                                print(f"LLM JSON ERROR: {e}")
                                continue
                else:
                    # Non-streaming: return full response
                    return res.json().get("response", "")
    except Exception as e:
        print("LLM ERROR:", e)
        if not stream:
            return ""


# -----------------------------------------------------
# 2. Strict prompt for stable short Vietnamese examples
# -----------------------------------------------------
def build_strict_seed_prompt(category_name: str):
    return f"""
Bạn là AI tạo ví dụ ngắn về giao dịch chi tiêu.

YÊU CẦU BẮT BUỘC:
- Tạo đúng 5 ví dụ.
- Mỗi ví dụ chỉ 2–4 từ.
- Không kể chuyện.
- Không hội thoại.
- Không mô tả dài.
- Không dấu chấm.
- Chỉ liệt kê gạch đầu dòng.

Ví dụ chuẩn:
- mua ao thun
- mua ao khoac
- mua quan jean

Bây giờ hãy tạo 5 ví dụ ngắn cho danh mục: "{category_name}"

Trả lời duy nhất theo định dạng:
- ví dụ 1
- ví dụ 2
- ví dụ 3
- ví dụ 4
- ví dụ 5
"""


# -----------------------------------------------------
# 3. Parse clean bullet-point output into list[str]
# -----------------------------------------------------
def parse_llm_examples(raw_text: str):
    """
    Convert bullet list from LLaMA into clean list of examples.
    """
    lines = []
    for line in raw_text.split("\n"):
        line = line.strip()
        if line.startswith("-"):
            example = line.replace("-", "").strip()
            if example:
                lines.append(example)
    return lines[:5]  # ensure at most 5 examples


# -----------------------------------------------------
# 4. High-level wrapper used by your seeding logic
# -----------------------------------------------------
def generate_seed_examples(category_name: str):
    """
    High-level helper for hybrid category seeding.
    - builds strict prompt
    - calls local LLaMA
    - parses bullet list
    """
    prompt = build_strict_seed_prompt(category_name)
    raw = call_local_llm(prompt)
    if not raw:
        return []

    examples = parse_llm_examples(raw)
    return examples