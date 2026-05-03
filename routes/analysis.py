"""Prediction and analysis routes"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from predictor import predict_next_month, analyze_spending_report, generate_budget_tips
from models.schemas import PredictionReport, AnalysisResponse

router = APIRouter(prefix="/api/analysis", tags=["Analysis & Predictions"])
logger = logging.getLogger(__name__)


# ============================================
# Prediction Endpoints
# ============================================

@router.post("/predict", summary="Predict next month's spending")
@router.get("/predict", summary="Predict next month's spending (test with mock data)")
async def predict_next_month_spending(report: Optional[PredictionReport] = None):
    """
    Predict next month's spending based on current financial report.
    
    - **POST**: Send full report data as JSON
    - **GET**: Use mock data for testing
    
    Returns streaming response with AI-generated predictions.
    """
    try:
        # Use mock data for GET requests or when report is None
        if report is None:
            report_dict = {
                "cashFlow": {"totalExpense": 1425, "transactionCount": 38},
                "availableBalance": 75,
                "expenseStructure": {
                    "categories": [
                        {
                            "categoryName": "badminton",
                            "amount": 600,
                            "percentage": 42.11,
                            "transactionCount": 15
                        },
                        {
                            "categoryName": "Food & Dining",
                            "amount": 475,
                            "percentage": 33.33,
                            "transactionCount": 12
                        }
                    ]
                },
                "periodComparison": {
                    "comparison": {
                        "expenseDelta": 50,
                        "expenseChangePercent": 3.6
                    }
                },
                "budgetProgress": {
                    "totalBudget": 1500,
                    "totalSpent": 1425,
                    "overallStatus": "ON_TRACK"
                }
            }
        else:
            report_dict = report.model_dump()
        
        return predict_next_month(report_dict)
    
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ============================================
# Analysis Endpoints
# ============================================

@router.post("/spending", summary="Analyze spending patterns")
async def analyze_spending(report: PredictionReport):
    """
    Analyze spending patterns and provide insights based on financial report.
    
    Returns detailed analysis of spending behavior, trends, and recommendations.
    """
    try:
        report_dict = report.model_dump()
        result = analyze_spending_report(report_dict)
        return result
    except Exception as e:
        logger.error(f"Error in spending analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/budget-tips", summary="Generate budget tips")
async def get_budget_tips(report: PredictionReport):
    """
    Generate personalized budget tips and recommendations.
    
    Returns actionable tips based on current spending patterns and budget status.
    """
    try:
        report_dict = report.model_dump()
        result = generate_budget_tips(report_dict)
        return result
    except Exception as e:
        logger.error(f"Error generating budget tips: {e}")
        raise HTTPException(status_code=500, detail=f"Budget tips generation failed: {str(e)}")
