"""RabbitMQ management and monitoring routes"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from rabbitmq_service import get_rabbitmq_service
from models.schemas import (
    RabbitMQMessage,
    RabbitMQQueueRequest,
    RabbitMQStatusResponse,
    RabbitMQPublishResponse,
)

router = APIRouter(prefix="/api/rabbitmq", tags=["RabbitMQ"])
logger = logging.getLogger(__name__)


# ============================================
# RabbitMQ Status & Monitoring
# ============================================

@router.get("/status", response_model=RabbitMQStatusResponse, summary="Check RabbitMQ status")
def get_rabbitmq_status():
    """
    Check RabbitMQ connection and consumer status.
    
    Returns connection state, consuming status, and configuration details.
    """
    try:
        rabbitmq = get_rabbitmq_service()
        
        is_connected = rabbitmq.connection and rabbitmq.connection.is_open
        is_consuming = rabbitmq._consuming
        
        return RabbitMQStatusResponse(
            connected=is_connected,
            consuming=is_consuming,
            exchange=rabbitmq.exchange,
            consumer_queue=rabbitmq.consumer_queue,
            producer_queue=rabbitmq.producer_queue,
            host=f"{rabbitmq.host}:{rabbitmq.port}"
        )
    except Exception as e:
        logger.error(f"Error checking RabbitMQ status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


# ============================================
# Message Publishing
# ============================================

@router.post("/publish", response_model=RabbitMQPublishResponse, summary="Publish message to RabbitMQ")
def publish_message(
    message: RabbitMQMessage,
    routing_key: Optional[str] = Query(None, description="Custom routing key")
):
    """
    Manually publish a message to RabbitMQ producer queue.
    
    **Example request body:**
    ```json
    {
        "status": "success",
        "action": "classify",
        "userId": "user123",
        "data": {...}
    }
    ```
    """
    try:
        rabbitmq = get_rabbitmq_service()
        message_dict = message.model_dump(exclude_none=True)
        
        success = rabbitmq.publish_message(message_dict, routing_key=routing_key)
        
        if success:
            return RabbitMQPublishResponse(
                status="success",
                message="Message published successfully",
                queue=rabbitmq.producer_queue
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to publish message")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error publishing message: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to publish: {str(e)}")


# ============================================
# Queue Management
# ============================================

@router.post("/queues", summary="Add consumer queue")
def add_consumer_queue(request: RabbitMQQueueRequest):
    """
    Dynamically add a new consumer queue to RabbitMQ.
    
    **Example:**
    ```json
    {
        "queue_name": "analytics.ai.queue",
        "routing_key": "analytics.ai.request"
    }
    ```
    
    If routing_key is not provided, it defaults to queue_name.
    """
    try:
        rabbitmq = get_rabbitmq_service()
        
        routing_key = request.routing_key or request.queue_name
        rabbitmq.add_consumer_queue(request.queue_name, routing_key)
        
        return {
            "status": "success",
            "message": f"Queue '{request.queue_name}' added successfully",
            "queue": request.queue_name,
            "routing_key": routing_key
        }
    
    except Exception as e:
        logger.error(f"Error adding consumer queue: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add queue: {str(e)}")
