
from operator import index
import json

def prepare_prediction_context(report):
    try:
        cash = report["cashFlow"]
        categories = report["expenseStructure"]["categories"]
        period = report["periodComparison"]
        budget = report["budgetProgress"]

        cat_lines = []
        for c in categories:
            name = c.get("categoryName") or "Unknown Category"
            cat_lines.append(
                f"- {name}: {c['amount']} ({c['percentage']}%), count={c['transactionCount']}"
            )

        summary = f"""
Tổng chi tháng hiện tại: {cash['totalExpense']}
Số giao dịch: {cash['transactionCount']}
Số dư còn lại: {report['availableBalance']}
Danh mục chi tiêu:
{chr(10).join(cat_lines)}

So sánh tháng trước:
- Tăng/giảm chi tiêu: {period['comparison']['expenseDelta']}
- Tăng/giảm %: {period['comparison']['expenseChangePercent']}

Tiến độ ngân sách:
Tổng ngân sách: {budget['totalBudget']}
Đã tiêu: {budget['totalSpent']}
Trạng thái chung: {budget['overallStatus']}
"""
        return summary.strip()
    except Exception as e:
        return f"(Error preparing data: {str(e)})"

def predict_next_month(report_data):
    from llm_client import call_local_llm

    # Step 1: prepare input context for the LLM
    prepared = prepare_prediction_context(report_data)

    # Step 2: load the prompt template
    with open("prompts/prediction_prompt.txt", "r", encoding="utf-8") as f:
        template = f.read()

    # Step 3: format the final prompt
    prompt = template.format(prepared_data=prepared)

    # Step 4: call LLaMA with streaming
    def generate():
        buffer = ""
        for chunk in call_local_llm(prompt, temperature=0.4, stream=True):
            if chunk:
                buffer += chunk
                # Send chunks at word boundaries when buffer reaches reasonable size
                if len(buffer) >= 100 and chunk in [' ', '\n', '.', '!', '?', '|', ')']:
                    yield f"data: {json.dumps({'text': buffer}, ensure_ascii=False)}\n\n"
                    buffer = ""
        
        # Send any remaining text
        if buffer:
            yield f"data: {json.dumps({'text': buffer}, ensure_ascii=False)}\n\n"
    
    from fastapi.responses import StreamingResponse
    return StreamingResponse(generate(), media_type="text/event-stream")

# {
#     "cashFlow": {
#         "totalIncome": 15000000,
#         "totalExpense": 9200000,
#         "netIncome": 5800000,
#         "transactionCount": 47
#     },
#     "availableBalance": 8650000,
#     "monthComparison": {
#         "previousMonthIncome": 14000000,
#         "previousMonthExpense": 8600000,
#         "previousMonthNet": 5400000,
#         "incomeChange": 1000000,
#         "expenseChange": 600000,
#         "netChange": 400000,
#         "incomeChangePercent": 7.14,
#         "expenseChangePercent": 6.98,
#         "netChangePercent": 7.40
#     },
#     "year": 2025,
#     "month": 11,
#     "generatedAt": "2025-11-23T16:06:32.302725"
# }
def prepare_analysis_context(report):
    try:
        cash = report["cashFlow"]
        balance = report["availableBalance"]
        month_comp = report["monthComparison"]
        categories = report["expenseStructure"]["categories"]

        # Format expense categories
        cat_lines = []
        for c in categories:
            name = c.get("categoryName") or "Unknown Category"
            cat_lines.append(
                f"- {name}: {c['amount']} ({c['percentage']}%), "
                f"giao dịch={c['transactionCount']}"
            )

        summary = f"""
BÁO CÁO CHI TIÊU THÁNG {report.get('month')}/{report.get('year')}:

1) Tổng quan chi tiêu:
- Tổng thu nhập: {cash['totalIncome']}
- Tổng chi tiêu: {cash['totalExpense']}
- Thu nhập ròng: {cash['netIncome']}
- Số dư khả dụng: {balance}
- Tổng giao dịch: {cash['transactionCount']}

2) Chi tiêu theo danh mục:
{chr(10).join(cat_lines)}

3) So sánh với tháng trước:
- Thay đổi thu nhập: {month_comp['incomeChange']} ({month_comp['incomeChangePercent']}%)
- Thay đổi chi tiêu: {month_comp['expenseChange']} ({month_comp['expenseChangePercent']}%)
- Thay đổi ròng: {month_comp['netChange']} ({month_comp['netChangePercent']}%)
"""
        return summary.strip()
    except Exception as e:
        return f"(Error preparing data: {str(e)})"


def analyze_spending_report(report):
    from llm_client import call_local_llm

    # FIXED: Correct function
    prepared = prepare_analysis_context(report)

    # Load prompt template
    with open("prompts/spending_analysis_prompt.txt", "r", encoding="utf-8") as f:
        template = f.read()

    prompt = template.format(prepared_data=prepared)

    # Stream output
    def generate():
        buffer = ""
        for chunk in call_local_llm(prompt, temperature=0.4, stream=True):
            if chunk:
                buffer += chunk
                if len(buffer) >= 100 and chunk in [' ', '\n', '.', '!', '?', ')']:
                    yield f"data: {json.dumps({'text': buffer}, ensure_ascii=False)}\n\n"
                    buffer = ""
        
        if buffer:
            yield f"data: {json.dumps({'text': buffer}, ensure_ascii=False)}\n\n"

    from fastapi.responses import StreamingResponse
    return StreamingResponse(generate(), media_type="text/event-stream")



{
  "cashFlow": {
    "totalIncome": 15000000,
    "totalExpense": 9200000,
    "netIncome": 5800000,
    "transactionCount": 47
  },

  "availableBalance": 8650000,

  "expenseStructure": {
    "totalExpense": 9200000,
    "categories": [
      {
        "categoryId": "f23be549-98a3-417a-b6e8-17f5ae4b22c3",
        "categoryName": "Badminton",
        "icon": "",
        "amount": 3500000,
        "percentage": 38.04,
        "transactionCount": 3
      },
      {
        "categoryId": "85e38348-bcac-45fe-9cc5-282788a533af",
        "categoryName": "Food & Dining",
        "icon": "dab-a",
        "amount": 2800000,
        "percentage": 30.43,
        "transactionCount": 7
      },
      {
        "categoryId": "e64fd287-f8ac-4516-9726-069be3ed294d",
        "categoryName": "Recreation",
        "icon": "",
        "amount": 2400000,
        "percentage": 26.09,
        "transactionCount": 2
      },
      {
        "categoryId": "e2b5a0e8-3e35-4a92-afe4-ffe7a111c25b",
        "categoryName": "Other",
        "icon": "",
        "amount": 500000,
        "percentage": 5.43,
        "transactionCount": 1
      }
    ],
    "periodStart": "2025-11-01T00:00:00",
    "periodEnd": "2025-11-30T23:59:59"
  },

  "budgetProgress": {
    "year": 2025,
    "month": 11,
    "totalBudget": 10000000,
    "totalSpent": 9200000,
    "totalRemaining": 800000,
    "overallProgress": 92,
    "overallStatus": "WARNING",

    "categoryBudgets": [
      {
        "categoryId": "f23be549-98a3-417a-b6e8-17f5ae4b22c3",
        "categoryName": "Badminton",
        "icon": "",
        "limitAmount": 3000000,
        "spentAmount": 3500000,
        "remaining": -500000,
        "progressPercentage": 117,
        "status": "OVER_BUDGET",
        "transactionCount": 3
      },
      {
        "categoryId": "85e38348-bcac-45fe-9cc5-282788a533af",
        "categoryName": "Food & Dining",
        "icon": "dab-a",
        "limitAmount": 3000000,
        "spentAmount": 2800000,
        "remaining": 200000,
        "progressPercentage": 93,
        "status": "WARNING",
        "transactionCount": 7
      },
      {
        "categoryId": "e64fd287-f8ac-4516-9726-069be3ed294d",
        "categoryName": "Recreation",
        "icon": "",
        "limitAmount": 2000000,
        "spentAmount": 2400000,
        "remaining": -400000,
        "progressPercentage": 120,
        "status": "OVER_BUDGET",
        "transactionCount": 2
      },
      {
        "categoryId": "e2b5a0e8-3e35-4a92-afe4-ffe7a111c25b",
        "categoryName": "Other",
        "icon": "",
        "limitAmount": 500000,
        "spentAmount": 500000,
        "remaining": 0,
        "progressPercentage": 100,
        "status": "ON_TRACK",
        "transactionCount": 1
      }
    ]
  }
}

def prepare_budget_context(report):
    try:
        budget = report["budgetProgress"]
        categories = budget["categoryBudgets"]
        categories = sorted(categories, key=lambda c: c['progressPercentage'], reverse=True)
        cat_lines = []
        for c in categories:
            name = c.get("categoryName") or "Unknown Category"
            cat_lines.append(
                f"- {name}: Ngân sách={c['limitAmount']}, Đã chi={c['spentAmount']}, "
                f"Còn lại={c['remaining']}, Tiến độ={c['progressPercentage']}%, "
                f"Trạng thái={c['status']}, Giao dịch={c['transactionCount']}"
            )
        
        summary = f"""
BÁO CÁO TIẾN ĐỘ NGÂN SÁCH THÁNG {budget.get('month')}/{budget.get('year')}:
- Tổng ngân sách: {budget['totalBudget']}
- Đã chi: {budget['totalSpent']}
- Còn lại: {budget['totalRemaining']}
- Trạng thái chung: {budget['overallStatus']}

Chi tiết theo danh mục:
{chr(10).join(cat_lines)}
"""
        return summary.strip()
    except Exception as e:
        return f"(Error preparing data: {str(e)})"
    
def generate_budget_tips(report):
    from llm_client import call_local_llm

    prepared = prepare_budget_context(report)

    with open("prompts/budget_analysis_prompt.txt", "r", encoding="utf-8") as f:
        template = f.read()

    prompt = template.format(prepared_data=prepared)

    def generate():
        buffer = ""
        for chunk in call_local_llm(prompt, temperature=0.4, stream=True):
            if chunk:
                buffer += chunk
                if len(buffer) >= 100 and chunk in [' ', '\n', '.', '!', '?', ')']:
                    yield f"data: {json.dumps({'text': buffer}, ensure_ascii=False)}\n\n"
                    buffer = ""
        
        if buffer:
            yield f"data: {json.dumps({'text': buffer}, ensure_ascii=False)}\n\n"

    from fastapi.responses import StreamingResponse
    return StreamingResponse(generate(), media_type="text/event-stream")