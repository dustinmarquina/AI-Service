
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
            - Tăng/giảm %: {period['comparison']['expenseChanPercent']}

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