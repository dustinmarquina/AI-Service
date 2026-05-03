from fastmcp import FastMCP
from predictor import (
    prepare_prediction_context,
    prepare_budget_context,
    prepare_analysis_context,
)
from mcp_service.dummy_llm import dummy_llm_response
import requests

mcp = FastMCP("Finance-ai")

@mcp.tool()
def navigate(payload: dict) -> str:
    """
    Navigate to a specific section of the user's financial dashboard.
    Expect Vietnamese input with a "section" key indicating where to navigate.
    Return a confirmation message in Vietnamese.
    """
    if "section" not in payload:
        return "Please specify the section you want to navigate to."

    section = payload["section"]
    # Here you would implement the actual navigation logic, e.g., by returning a URL or triggering a frontend action.
    return f"Đang điều hướng đến phần {section} của bảng điều khiển tài chính của bạn."


@mcp.tool()
def create_transaction(payload: dict) -> str:
    """
    Expect Vietnamese input
    Create a new financial transaction by calling the Spring Boot API if the user types either amount or description.
    If either amount or description is missing, ask the user to provide the missing information before making the API call.
    Return the raw JSON response from the API call as-is without any modification.
    """
    relay_url = "https://69b2bf5be06ef68ddd962420.mockapi.io/api/v1/transaction"
    action = "NAVIGATE"
    if "amount" not in payload:
        return "Please provide the transaction amount."
    if "description" not in payload:
        return "Please provide the transaction description."

    # call the API to create the transaction
    try: 
        response = requests.post(relay_url, json=payload)
        if response.status_code in [200, 201]:
            return f"{{\"status\": \"success\", \"message\": \"Transaction created successfully\", \"data\": {response.json()}}}"
        else:
            return f"{{\"status\": \"error\", \"message\": \"Failed to create transaction. Status: {response.status_code}, Response: {response.text}\"}}"
    except Exception as e:
        return f"{{\"status\": \"error\", \"message\": \"Exception occurred while creating transaction: {str(e)}\"}}"   
    # import requests

    # base_url = "{{baseUrl}}"
    # url = f"{base_url}/transactions"
    # response = requests.post(url, json=payload)


    # if response.status_code == 200 or response.status_code == 201:
    #     return f"Transaction created successfully: {response.json()}"
    # else:
    #     return f"Failed to create transaction. Status: {response.status_code}, Response: {response.text}"
    # add category name "programming" to the payload for testing

# mock response for testing without actual API call
#     {
#     "id": "40cad627-00f7-4a6c-a91d-16b998cf55f5",
#     "amount": 10,
#     "type": "EXPENSE",
#     "categoryId": "d6b78acc-0fa2-48ae-b0e0-268b210c2d55",
#     "categoryName": "string",
#     "description": "string",
#     "walletId": "6c1c6812-04f1-4965-8ba6-8af09bcd033c",
#     "userId": "829e15ae-eb6b-4226-a040-f636cf427fd9",
#     "transactionDate": "2026-03-11T12:59:11.865Z"
# }
    # response = {
    #     "id": "40cad627-00f7-4a6c-a91d-16b998cf55f5",
    #     "amount": payload["amount"],
    #     "type": "EXPENSE",
    #     "categoryId": "d6b78acc-0fa2-48ae-b0e0-268b210c2d55",
    #     "categoryName": "programming",
    #     "description": payload["description"],
    #     "walletId": "6c1c6812-04f1-4965-8ba6-8af09bcd033c",
    #     "userId": "829e15ae-eb6b-4226-a040-f636cf427fd9",
    #     "transactionDate": "2026-03-11T12:59:11.865Z",
    #     "redirect_url": redirect_url,
    #     "action": action
    # }
    # return f"{{\"status\": \"success\", \"message\": \"Transaction created successfully (simulated)\", \"data\": {None}}}"


@mcp.tool()
def analyze_spending(report: dict) -> str:
    """
    Analyze monthly spending report and return financial advice.
    """
    prepared = prepare_analysis_context(report)
    return dummy_llm_response(prepared)

@mcp.tool()
def generate_budget_tips(report: dict) -> str:
    """
    Analyze budget progress and generate actionable tips.
    """
    prepared = prepare_budget_context(report)
    return dummy_llm_response(prepared)


@mcp.tool()
def predict_next_month(report: dict) -> str:
    """
    Predict next month spending based on current report.
    """
    prepared = prepare_prediction_context(report)
    return dummy_llm_response(prepared)

@mcp.tool()
def get_status(payload: dict | None = None) -> str:
    """
    Get current status of the MCP server.
    """
    return "Finance-ai MCP server is running and ready to process requests though it's unstable currently, beware."


if __name__ == "__main__":
    mcp.run()

        