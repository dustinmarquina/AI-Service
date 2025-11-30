"""
Message handlers for processing RabbitMQ messages
Handles requests from Spring Boot microservice
"""
import logging
from typing import Dict, Any, Optional
from transaction_classifying import categorizeItem, addCategoryExample, addCustomCategory, initUserCategory, addCategoryExampleByCatgoryId
from tx_sandbox import clean_example_text

logger = logging.getLogger(__name__)


def handle_category_event(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Handle category events from category.ai.queue
    
    Expected message format:
    {
        "eventType": "CREATED" | "UPDATED" | "DELETED",
        "categoryId": "uuid",
        "userId": "uuid",
        "categoryName": "Food & Dining",
        "icon": "icon-name",
        "limitAmount": 500.0,
        "spentAmount": 0,
        "timestamp": [2025, 11, 24, 22, 30, 43, 297742000]
    }
    """
    try:
        event_type = message.get("eventType")
        category_id = message.get("categoryId")
        user_id = message.get("userId")
        category_name = message.get("categoryName")

        logger.info(f"🏷️  Category Event: {event_type} | Category: {category_name} | User: {user_id}")
        
        if event_type == "CREATED":
            # Initialize user category if first category, or add custom category
            logger.info(f"✅ Category CREATED: {category_name} (ID: {category_id})")
            addCustomCategory(userId=user_id, categoryId=category_id, categoryName=category_name)
            
            # You can add custom logic here, e.g.:
            # - Initialize user categories in ML model
            # - Store category mapping
            # - Train model with new category
            
            return {
                "status": "success",
                "eventType": event_type,
                "categoryId": category_id,
                "message": f"Category '{category_name}' created successfully"
            }
        
        elif event_type == "UPDATED":
            logger.info(f"🔄 Category UPDATED: {category_name} (ID: {category_id})")
            
            # Handle category update logic
            # - Update category name in ML model
            # - Retrain if needed
            
            return {
                "status": "success",
                "eventType": event_type,
                "categoryId": category_id,
                "message": f"Category '{category_name}' updated successfully"
            }
        
        elif event_type == "DELETED":
            logger.info(f"🗑️  Category DELETED: {category_name} (ID: {category_id})")
            
            # Handle category deletion
            # - Remove from ML model
            # - Clean up training data
            
            return {
                "status": "success",
                "eventType": event_type,
                "categoryId": category_id,
                "message": f"Category '{category_name}' deleted successfully"
            }
        
        else:
            return {
                "status": "error",
                "message": f"Unknown eventType: {event_type}"
            }
    
    except Exception as e:
        logger.error(f"❌ Error handling category event: {e}")
        return {
            "status": "error",
            "message": f"Failed to process category event: {str(e)}"
        }


def handle_transaction_event(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Handle transaction events from transaction.ai.queue
    
    Expected message format from Spring Boot:
    {
        "eventType": "CREATED" | "UPDATED" | "DELETED",
        "transactionId": "uuid",
        "userId": "uuid",
        "walletId": "uuid",
        "categoryId": "uuid",
        "description": "grab di truong",
        "amount": 50000.0,
        "timestamp": [2025, 11, 25, 13, 57, 43, 408812000]
    }
    """
    try:
        event_type = message.get("eventType")
        transaction_id = message.get("transactionId")
        user_id = message.get("userId")
        wallet_id = message.get("walletId")
        category_id = message.get("categoryId")
        description = message.get("description")
        amount = message.get("amount")
        
        logger.info(f"💳 Transaction Event: {event_type} | User: {user_id} | Description: {description}")
        
        if event_type == "CREATED":
            # Classify the transaction description
            if not description or not user_id:
                return {
                    "status": "error",
                    "eventType": event_type,
                    "message": "Missing 'description' or 'userId' for classification"
                }
            
            logger.info(f"🔍 Classifying transaction: {description}")
            result = categorizeItem(userId=user_id, item=description)
            
            if category_id and result != category_id:
                addCategoryExampleByCatgoryId(userId=user_id, categoryId=category_id, example=description)

            return {
                "status": "success",
                "transactionId": transaction_id,
                "userId": user_id,
                "walletId": wallet_id,
                "categoryId": result,
                "message": f"Transaction classified successfully"
            }
        
        elif event_type == "UPDATED":
            # Re-classify if description changed
            logger.info(f"🔄 Transaction UPDATED: {description} (ID: {transaction_id})")
            addCategoryExample(userId=user_id, categoryId=category_id, example=description)
            return {
                "status": "success",
                "eventType": event_type,
                "transactionId": transaction_id,
                "categoryId": category_id,
                "message": f"Transaction updated successfully"
            }
        
        elif event_type == "DELETED":
            logger.info(f"🗑️  Transaction DELETED: {description} (ID: {transaction_id})")
            
            # Handle transaction deletion if needed
            # - Remove from training data
            # - Update statistics
            
            return {
                "status": "success",
                "eventType": event_type,
                "transactionId": transaction_id,
                "message": f"Transaction deleted successfully"
            }
        
        else:
            return {
                "status": "error",
                "message": f"Unknown eventType: {event_type}"
            }
    
    except Exception as e:
        logger.error(f"❌ Error handling transaction event: {e}")
        return {
            "status": "error",
            "message": f"Failed to process transaction event: {str(e)}"
        }


def handle_message(message: Dict[str, Any], queue_name: str = None, routing_key: str = None) -> Optional[Dict[str, Any]]:
    """
    Process incoming messages from Spring Boot
    
    Expected message format:
    {
        "action": "classify" | "add_example" | "add_category" | "init_user" | "clean_text",
        "userId": "user123",
        "text": "grab di truong",
        "categoryId": "1",  // for add_example
        "categoryName": "Food"  // for add_category
    }
    
    Args:
        message: The message dict from RabbitMQ
        queue_name: Optional queue name where message came from
        routing_key: Optional routing key used
    
    Returns response:
    {
        "status": "success" | "error",
        "action": "classify",
        "data": {...},
        "message": "..."
    }
    """
    try:
        # Log which queue this came from
        if queue_name:
            logger.info(f"📥 Message from queue: {queue_name}")
        
        # ====== CATEGORY QUEUE HANDLER ======
        # category.ai.queue from category.ai.exchange (routing key might be category.ai.event or category.ai.queue)
        if queue_name == "category.ai.queue" or (routing_key and "category.ai" in routing_key):
            return handle_category_event(message)
        
        # ====== TRANSACTION QUEUE HANDLER ======
        # transaction.ai.request.queue from transaction.ai.exchange
        elif queue_name == "transaction.ai.request.queue" or (routing_key and routing_key == "transaction.ai.request"):
            return handle_transaction_event(message)
        elif queue_name and ("transaction.ai" in queue_name or routing_key and "transaction.ai" in routing_key):
            return handle_transaction_event(message)
        
        # ====== LEGACY/DEFAULT ACTION-BASED HANDLER ======
        action = message.get("action")
        user_id = message.get("userId")
        
        if not action:
            return {
                "status": "error",
                "message": "Missing 'action' field in message"
            }
        
        logger.info(f"Processing action: {action} for user: {user_id}")
        
        # ====== CLASSIFY ACTION ======
        if action == "classify":
            text = message.get("text")
            if not text or not user_id:
                return {
                    "status": "error",
                    "action": action,
                    "message": "Missing 'text' or 'userId' for classification"
                }
            
            result = categorizeItem(userId=user_id, item=text)
            
            return {
                "status": "success",
                "action": action,
                "userId": user_id,
                "data": result,
                "message": "Classification completed successfully"
            }
        
        # ====== ADD EXAMPLE ACTION ======
        elif action == "add_example":
            text = message.get("text")
            category_id = message.get("categoryId")
            
            if not text or not user_id or not category_id:
                return {
                    "status": "error",
                    "action": action,
                    "message": "Missing 'text', 'userId', or 'categoryId'"
                }
            
            result = addCategoryExample(userId=user_id, categoryId=category_id, example=text)
            
            return {
                "status": "success",
                "action": action,
                "userId": user_id,
                "data": result,
                "message": "Example added successfully"
            }
        
        # ====== ADD CUSTOM CATEGORY ACTION ======
        elif action == "add_category":
            category_name = message.get("categoryName")
            
            if not user_id or not category_name:
                return {
                    "status": "error",
                    "action": action,
                    "message": "Missing 'userId' or 'categoryName'"
                }
            
            result = addCustomCategory(userId=user_id, categoryName=category_name)
            
            return {
                "status": "success",
                "action": action,
                "userId": user_id,
                "data": result,
                "message": f"Category '{category_name}' added successfully"
            }
        
        # ====== INIT USER ACTION ======
        elif action == "init_user":
            if not user_id:
                return {
                    "status": "error",
                    "action": action,
                    "message": "Missing 'userId'"
                }
            
            initUserCategory(userId=user_id)
            
            return {
                "status": "success",
                "action": action,
                "userId": user_id,
                "message": "User categories initialized successfully"
            }
        
        # ====== CLEAN TEXT ACTION ======
        elif action == "clean_text":
            text = message.get("text")
            if not text:
                return {
                    "status": "error",
                    "action": action,
                    "message": "Missing 'text' field"
                }
            
            result = clean_example_text(text)
            
            return {
                "status": "success",
                "action": action,
                "data": result,
                "message": "Text cleaned successfully"
            }
        
        # ====== UNKNOWN ACTION ======
        else:
            return {
                "status": "error",
                "action": action,
                "message": f"Unknown action: {action}"
            }
    
    except Exception as e:
        logger.error(f"❌ Error handling message: {e}", exc_info=True)
        return {
            "status": "error",
            "action": message.get("action"),
            "message": f"Internal error: {str(e)}"
        }
