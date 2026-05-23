from dotenv import load_dotenv
from pymongo import response
load_dotenv(override=True)
# from mcp_service.tools.db_connect import connect_to_postgres
from langchain_community.utilities import SQLDatabase
import os


def test_connection():
    try:
        # get from .env file
        db = SQLDatabase.from_uri(os.getenv("POSTGRES_URI", "").strip())
        # print("✅ Connected successfully")
        # try call API POST https://api-budget-tracker.manportfolio.id.vn/api/v1/categories with Authorization Bearer token from .env file
        # token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhZjFlYjIwYi01NGU1LTQ5MTAtYjk3NC02MDE1MTE4NWU0OGYiLCJhY2NvdW50SWQiOiIzNTcwMmZiMy04NjYyLTRmYmMtOGZiNC00Mjc3YTUwNGM3YmEiLCJlbWFpbCI6ImRlbW8wMDFAZ21haWwuY29tIiwicm9sZXMiOiJST0xFX1VTRVIiLCJpYXQiOjE3Nzc4Nzk4MTIsImV4cCI6MTc3Nzk2NjIxMn0.mcAePJb40MFfMYKuXWk_jBSs2KhRWpeOkWguyfb90ME"
        # import httpx
        # response = httpx.post(
        #     "https://api-budget-tracker.manportfolio.id.vn/api/v1/transactions",
        #     json={
        #         "userId": "af1eb20b-54e5-4910-b974-60151185e48f",
        #         "amount": 1000,  # dummy amount for category creation
        #         "description": "Category creation via chatbot",
        #         "type": "EXPENSE",
        #         "categoryName": "Test Category",
        #         # "trấnc"
                
        #     },
        #     headers={
        #         "Authorization": f"Bearer {token}"
        #     }
        # )

        # response:
#         {
#     "id": "0478e2eb-758c-47ad-b012-2a2cca4e66ed",
#     "amount": 15,
#     "type": "EXPENSE",
#     "categoryId": "73a043c5-7654-454a-bb5e-4fe7e3ac5176",
#     "categoryName": "string3",
#     "description": "string",
#     "imageUrl": "https://minio.manportfolio.id.vn/budget-tracker/transactions/af1eb20b-54e5-4910-b974-60151185e48f/049db6f7-4193-4945-a9bc-4d4f871e1c81.jpeg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=Vq5xelHh3JNsscSuBZQi%2F20260502%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260502T151523Z&X-Amz-Expires=600&X-Amz-SignedHeaders=host&X-Amz-Signature=96a1e0085a91b5a8db2a2ceeb24faea525b470c80a2a783fc35e89dbaa334ad8",
#     "walletId": "0faffd3a-c58c-4323-a112-9b0ab2f2f7e4",
#     "userId": "af1eb20b-54e5-4910-b974-60151185e48f",
#     "transactionDate": "2026-03-11T12:59:11.865Z"
# }
        # print("API Response:", response.json())


        # import os
        # print("\n--- POSTGRES_URI FROM ENV ---")
        # print(os.getenv("POSTGRES_URI"))
        # print("\n--- ALL SCHEMAS ---")
        # print(db.run("""
        # SELECT schema_name 
        # FROM information_schema.schemata;
        # """))

        # print("\n--- ALL TABLES ---")
        # print(db.run("""
        # SELECT table_schema, table_name 
        # FROM information_schema.tables
        # WHERE table_type='BASE TABLE';
        # """))

        # print("\n--- PUBLIC TABLES ONLY ---")
        # print(db.run("""
        # SELECT * FROM users;
        # """))    

        # print("\n--- CURRENT DATABASE ---")
        # print(db.run("SELECT current_database();"))

        # print("\n--- CURRENT USER ---")
        # print(db.run("SELECT current_user;"))

        print(f"Dialect: {db.dialect}")
        print(f"Available tables: {db.get_usable_table_names()}")
        print(f'Table info: {db.get_table_info(["wallets"])}')
        #execute a simple query
        result = db.run("SELECT id, name FROM wallets WHERE user_id = 'af1eb20b-54e5-4910-b974-60151185e48f';")

        print(f"Query Result: {result}")

        # print("✅ Connected successfully")

        #test chat groq with a simple prompt
        # from agents.orchestrator.llm import get_9router_llm
        # llm = get_9router_llm()
        # response = llm.invoke("what time is it?")
        # print("LLM Response:", response.content)

        # DEBUG: make a raw HTTP request to determine the exact endpoint and response
        # try:
        #     import httpx
        #     client_obj = getattr(llm, 'client', None)
        #     base = None
        #     api_key = os.getenv('9router_api_key') or os.getenv('GROQ_API_KEY')
        #     if client_obj and hasattr(client_obj, '_client'):
        #         base = getattr(client_obj._client, 'base_url', None)
        #     print("[debug] groq client base:", base)
        #     if base:
        #         candidates = [
        #             '/v1/chat/completions',
        #             '/openai/v1/chat/completions',
        #             '/api/v1/chat/completions',
        #         ]
        #         header_variants = [
        #             {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
        #             {'x-api-key': api_key, 'Content-Type': 'application/json'},
        #         ]
        #         payload = {
        #             'model': llm.model_name,
        #             'messages': [
        #                 {'role': 'user', 'content': [{'type': 'text', 'text': 'What is 2 + 2?'}]}
        #             ],
        #         }
        #         for path in candidates:
        #             url = str(base).rstrip('/') + path
        #             for headers in header_variants:
        #                 try:
        #                     r = httpx.post(url, json=payload, headers=headers, timeout=15.0)
        #                     print('[debug] POST', url, 'headers=', list(headers.keys()), '->', r.status_code)
        #                     text = r.text
        #                     print('[debug] resp starts with:', text[:200])
        #                 except Exception as e:
        #                     print('[debug] request to', url, 'failed:', e)
        # except Exception as e:
        #     print('[debug] direct request failed:', e)

        # tables = db.get_usable_table_names()
        # print("Tables:", tables)

    except Exception as e:
        print("❌ Connection failed:", e)


if __name__ == "__main__":
    test_connection()
