import time

def dummy_llm_response(prompt: str):
    # simulate thinking
    time.sleep(0.5)

    if "TIẾN ĐỘ NGÂN SÁCH" in prompt:
        return "Bạn đang vượt ngân sách ở danh mục Ăn uống. Hãy cân nhắc giảm chi tiêu."

    if "BÁO CÁO CHI TIÊU" in prompt:
        return "Chi tiêu tháng này tăng nhẹ so với tháng trước, chủ yếu do danh mục Mua sắm."

    return "Dự đoán tháng tới chi tiêu sẽ ổn định nếu duy trì thói quen hiện tại."