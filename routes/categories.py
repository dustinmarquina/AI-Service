"""Category and transaction classification routes"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from transaction_classifying import (
    addCategoryExampleByCatgoryId,
    initUserCategory,
    categorizeItem,
    addCustomCategory,
    modelize,
    clean_example_text,
    resetCentroids,
    deleteCategoryByUserId,
)
from llm_client import build_strict_seed_prompt, call_local_llm
from models.schemas import (
    CategoryRequest,
    CategorizeRequest,
    AddExampleRequest,
    GenerateSeedExamplesRequest,
    SeedExamplesResponse,
    ModelizeRequest,
    CleanTextResponse,
)

router = APIRouter(prefix="/api/categories", tags=["Categories"])
logger = logging.getLogger(__name__)


# ============================================
# Category Management
# ============================================

@router.post("/custom", summary="Create a custom category")
def create_custom_category(request: CategoryRequest):
    """Create a new custom category for a user"""
    try:
        result = addCustomCategory(userId=request.user_id, categoryName=request.category_name)
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Error creating custom category: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/init/{user_id}", summary="Initialize user categories")
def initialize_user_categories(user_id: str):
    """Initialize default categories for a user"""
    try:
        initUserCategory(userId=user_id)
        return {"status": "success", "message": f"Categories initialized for user {user_id}"}
    except Exception as e:
        logger.error(f"Error initializing user categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{user_id}", summary="Delete user categories")
def delete_user_categories(user_id: str):
    """Delete all categories for a user"""
    try:
        result = deleteCategoryByUserId(userId=user_id)
        return {"status": "success", "message": "Categories deleted", "data": result}
    except Exception as e:
        logger.error(f"Error deleting user categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{user_id}/reset", summary="Reset category centroids")
def reset_category_centroids(user_id: str):
    """Reset the category centroids for a user"""
    try:
        resetCentroids(userId=user_id)
        return {"status": "success", "message": "Centroids reset successfully"}
    except Exception as e:
        logger.error(f"Error resetting centroids: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Transaction Categorization
# ============================================

@router.post("/{user_id}/categorize", summary="Categorize a transaction")
def categorize_transaction(user_id: str, request: CategorizeRequest):
    """Categorize a transaction item for a user"""
    try:
        result = categorizeItem(userId=user_id, item=request.item)
        return result
    except Exception as e:
        logger.error(f"Error categorizing item: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{user_id}/examples", summary="Add example to category")
def add_category_example(user_id: str, request: AddExampleRequest):
    """Add an example to a specific category"""
    try:
        result = addCategoryExampleByCatgoryId(
            userId=user_id,
            categoryId=request.category_id,
            example=request.example
        )
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Error adding example: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Text Processing
# ============================================

@router.get("/clean-text", summary="Clean and normalize text")
def clean_text(item: str = Query(..., description="Text to clean")):
    """Clean and normalize transaction text"""
    try:
        cleaned = clean_example_text(item)
        return CleanTextResponse(cleaned_text=cleaned)
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract-amount", summary="Extract amount from text")
def extract_amount(request: ModelizeRequest):
    """Extract amount and other information from transaction text"""
    try:
        result = modelize(request.text, userId=request.user_id)
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Error extracting amount: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Seed Data Generation
# ============================================

@router.post("/generate-seeds", summary="Generate seed examples using LLM")
def generate_seed_examples(request: GenerateSeedExamplesRequest):
    """Generate seed examples for a category using LLM"""
    try:
        prompt = build_strict_seed_prompt(request.category_name)
        response = call_local_llm(prompt, temperature=0.3)
        return SeedExamplesResponse(category_name=request.category_name, examples=response)
    except Exception as e:
        logger.error(f"Error generating seed examples: {e}")
        raise HTTPException(status_code=500, detail=str(e))
