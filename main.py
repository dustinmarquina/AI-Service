from typing import Union
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from tx_sandbox import modelize, clean_example_text
from transaction_classifying import addCategoryExample, initUserCategory, categorizeItem, addCustomCategory
from predictor import predict_next_month
from rabbitmq_service import get_rabbitmq_service
from message_handler import handle_message

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage RabbitMQ connection lifecycle"""
    rabbitmq = get_rabbitmq_service()
    
    try:
        # Startup
        logger.info("🚀 Starting FastAPI application...")
        rabbitmq.connect()
        rabbitmq.set_message_handler(handle_message)
        
        # Example: Dynamically add more queues programmatically
        # rabbitmq.add_consumer_queue("analytics.ai.queue", "analytics.ai.request")
        # rabbitmq.add_consumer_queue("report.ai.queue", "report.ai.request")
        
        rabbitmq.start_consuming_background()
        logger.info("✅ RabbitMQ consumer started in background")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to start RabbitMQ: {e}")
        logger.info("⚠️ Continuing without RabbitMQ...")
        yield
        
    finally:
        # Shutdown
        logger.info("🛑 Shutting down FastAPI application...")
        try:
            rabbitmq.stop_consuming()
            rabbitmq.disconnect()
        except:
            pass


app = FastAPI(lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/items")
def add_custom_category(userId: str, categoryName: str):
    return addCustomCategory(userId=userId, categoryName=categoryName)

@app.post("/items/{item_id}")
def create_item(item_id: int, q: Union[str, None] = None):
    initUserCategory(userId=str(item_id))

@app.post("/item/categorize/{user_id}")
def categorize(user_id: int, item: str, q: Union[str, None] = None):
    return categorizeItem(userId=str(user_id), item=item)

@app.post("/item/add/{user_id}")
def addExample(user_id: int, categoryId: str, item: str, q: Union[str, None] = None):
    return addCategoryExample(userId=str(user_id), categoryId=categoryId, example=item)

@app.delete("/items/{item_id}")
def delete_item(item_id: int):
    from transaction_classifying import deleteCategoryByUserId
    result = deleteCategoryByUserId(userId=str(item_id))
    return {"item_id": item_id, "result": result}

@app.get("/extract_amount/")
def extract_amount_endpoint(text: str, userId: str):
    return modelize(text, userId=userId)

@app.post("/generate_seed_examples/")
def generate_seed_examples(category_name: str):
    from llm_client import build_strict_seed_prompt, call_local_llm
    prompt = build_strict_seed_prompt(category_name)
    response = call_local_llm(prompt, temperature=0.3)
    return {"category_name": category_name, "examples": response}

@app.api_route("/predict_next_month", methods=["GET", "POST", "OPTIONS"])
@app.api_route("/predict_next_month/", methods=["GET", "POST", "OPTIONS"])
async def predict_next_month_endpoint(report: dict = None):
    """
    Endpoint for LLM prediction streaming
    Use POST with JSON body for full report data
    Use GET for testing (will use mock data)
    """
    if report is None:
        # Mock data for GET requests
        report = {
            "cashFlow": {"totalExpense": 1425, "transactionCount": 38},
            "availableBalance": 75,
            "expenseStructure": {
                "categories": [
                    {"categoryName": "badminton", "amount": 600, "percentage": 42.11, "transactionCount": 15},
                    {"categoryName": "Food & Dining", "amount": 475, "percentage": 33.33, "transactionCount": 12}
                ]
            },
            "periodComparison": {
                "comparison": {"expenseDelta": 50, "expenseChangePercent": 3.6}
            },
            "budgetProgress": {
                "totalBudget": 1500,
                "totalSpent": 1425,
                "overallStatus": "ON_TRACK"
            }
        }
    return predict_next_month(report)

# spending analysis endpoint
@app.post("/analysis/spending")
async def analyze_spending(report: dict):
    from predictor import analyze_spending_report
    return analyze_spending_report(report)

@app.post("/generate/budget-tips")
async def budget_tips(report: dict):
    from predictor import generate_budget_tips
    return generate_budget_tips(report)


# ============================================
# RabbitMQ Management Endpoints
# ============================================

@app.get("/rabbitmq/status")
def rabbitmq_status():
    """Check RabbitMQ connection status"""
    rabbitmq = get_rabbitmq_service()
    
    is_connected = rabbitmq.connection and rabbitmq.connection.is_open
    is_consuming = rabbitmq._consuming
    
    return {
        "connected": is_connected,
        "consuming": is_consuming,
        "exchange": rabbitmq.exchange,
        "consumer_queue": rabbitmq.consumer_queue,
        "producer_queue": rabbitmq.producer_queue,
        "host": f"{rabbitmq.host}:{rabbitmq.port}"
    }


@app.post("/rabbitmq/publish")
def publish_to_rabbitmq(message: dict, routing_key: str = None):
    """
    Manually publish a message to RabbitMQ producer queue
    
    Example request body:
    {
        "status": "success",
        "action": "classify",
        "userId": "user123",
        "data": {...}
    }
    """
    rabbitmq = get_rabbitmq_service()
    
    success = rabbitmq.publish_message(message, routing_key=routing_key)
    
    if success:
        return {
            "status": "success",
            "message": "Message published successfully",
            "queue": rabbitmq.producer_queue
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to publish message")


@app.post("/rabbitmq/add-queue")
def add_consumer_queue(queue_name: str, routing_key: str = None):
    """
    Dynamically add a new consumer queue
    
    Example: POST /rabbitmq/add-queue?queue_name=analytics.ai.queue&routing_key=analytics.ai.request
    """
    rabbitmq = get_rabbitmq_service()
    
    try:
        rabbitmq.add_consumer_queue(queue_name, routing_key)
        return {
            "status": "success",
            "message": f"Queue '{queue_name}' added successfully",
            "queue": queue_name,
            "routing_key": routing_key or queue_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add queue: {str(e)}")