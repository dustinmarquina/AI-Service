"""
RabbitMQ Service - Consumer and Producer
Handles message queue operations for FastAPI <-> Spring Boot communication
"""
import os
import json
import logging
import threading
import pika
from typing import Callable, Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


class RabbitMQService:
    """RabbitMQ service for consuming and producing messages"""
    
    def __init__(self):
        # Connection settings
        self.host = os.getenv("RABBITMQ_HOST", "localhost")
        self.port = int(os.getenv("RABBITMQ_PORT", "5672"))
        self.username = os.getenv("RABBITMQ_USER", "guest")
        self.password = os.getenv("RABBITMQ_PASSWORD", "guest")
        self.vhost = os.getenv("RABBITMQ_VHOST", "/")
        
        # Exchanges
        self.exchange = os.getenv("RABBITMQ_EXCHANGE", "category.ai.exchange")
        self.transaction_exchange = os.getenv("RABBITMQ_TRANSACTION_EXCHANGE", "transaction.ai.exchange")
        
        # Store all exchanges
        self.exchanges = [self.exchange, self.transaction_exchange]
        
        # Consumer queues (receive from Spring Boot)
        self.consumer_queue = os.getenv("RABBITMQ_CONSUMER_QUEUE", "python.classify.request")
        self.consumer_routing_key = os.getenv("RABBITMQ_CONSUMER_ROUTING_KEY", "transaction.classify.request")
        
        # Additional consumer queues (comma-separated in .env)
        additional_queues = os.getenv("RABBITMQ_ADDITIONAL_QUEUES", "")
        additional_keys = os.getenv("RABBITMQ_ADDITIONAL_ROUTING_KEYS", "")
        
        self.consumer_queues = [self.consumer_queue]
        self.consumer_routing_keys = [self.consumer_routing_key]
        
        if additional_queues:
            self.consumer_queues.extend([q.strip() for q in additional_queues.split(",") if q.strip()])
        if additional_keys:
            self.consumer_routing_keys.extend([k.strip() for k in additional_keys.split(",") if k.strip()])
        
        # Producer queue (send to Spring Boot)
        self.producer_queue = os.getenv("RABBITMQ_PRODUCER_QUEUE", "python.classify.response")
        self.producer_routing_key = os.getenv("RABBITMQ_PRODUCER_ROUTING_KEY", "transaction.classify.response")
        
        # Connection objects
        self.connection: Optional[pika.BlockingConnection] = None
        self.channel: Optional[pika.channel.Channel] = None
        self.consumer_thread: Optional[threading.Thread] = None
        self.message_handler: Optional[Callable] = None
        self._consuming = False
        
        # Dynamic queue registry: {queue_name: (routing_key, handler)}
        self._dynamic_queues: Dict[str, tuple] = {}
        
    def connect(self):
        """Establish connection to RabbitMQ"""
        try:
            credentials = pika.PlainCredentials(self.username, self.password)
            parameters = pika.ConnectionParameters(
                host=self.host,
                port=self.port,
                virtual_host=self.vhost,
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300
            )
            
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            
            # Declare all exchanges
            for exchange in self.exchanges:
                self.channel.exchange_declare(
                    exchange=exchange,
                    exchange_type='topic',
                    durable=True
                )
                logger.info(f"📡 Declared exchange: {exchange}")
            
            # Declare and bind consumer queues
            # category.ai.queue -> category.ai.exchange
            self.channel.queue_declare(queue=self.consumer_queue, durable=True)
            self.channel.queue_bind(
                exchange=self.exchange,
                queue=self.consumer_queue,
                routing_key=self.consumer_routing_key
            )
            logger.info(f"📥 Consumer Queue 1: {self.consumer_queue} -> {self.exchange}")
            
            # transaction.ai.request.queue -> transaction.ai.exchange
            if len(self.consumer_queues) > 1:
                for i in range(1, len(self.consumer_queues)):
                    queue = self.consumer_queues[i]
                    routing_key = self.consumer_routing_keys[i] if i < len(self.consumer_routing_keys) else queue
                    
                    self.channel.queue_declare(queue=queue, durable=True)
                    self.channel.queue_bind(
                        exchange=self.transaction_exchange,
                        queue=queue,
                        routing_key=routing_key
                    )
                    logger.info(f"📥 Consumer Queue {i+1}: {queue} -> {self.transaction_exchange} (routing: {routing_key})")
            
            # Declare producer queue (transaction.ai.reply.queue -> transaction.ai.exchange)
            self.channel.queue_declare(
                queue=self.producer_queue,
                durable=True
            )
            
            # Bind producer queue to transaction exchange
            self.channel.queue_bind(
                exchange=self.transaction_exchange,
                queue=self.producer_queue,
                routing_key=self.producer_routing_key
            )
            
            logger.info(f"✅ Connected to RabbitMQ at {self.host}:{self.port}")
            logger.info(f"📤 Producer Queue: {self.producer_queue}")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to RabbitMQ: {e}")
            raise
    
    def disconnect(self):
        """Close RabbitMQ connection"""
        self._consuming = False
        
        if self.channel and self.channel.is_open:
            self.channel.close()
            
        if self.connection and self.connection.is_open:
            self.connection.close()
            
        logger.info("🔌 Disconnected from RabbitMQ")
    
    def publish_message(self, message: Dict[str, Any], routing_key: Optional[str] = None, exchange: Optional[str] = None) -> bool:
        """
        Publish a message to the producer queue
        
        Args:
            message: Dictionary to send as JSON
            routing_key: Optional routing key (defaults to producer_routing_key)
            exchange: Optional exchange (defaults to transaction_exchange)
        
        Returns:
            True if published successfully
        """
        if not self.channel or self.channel.is_closed:
            logger.error("❌ Channel not open. Reconnecting...")
            self.connect()
        
        try:
            message_body = json.dumps(message, ensure_ascii=False)
            
            self.channel.basic_publish(
                exchange=exchange or self.transaction_exchange,
                routing_key=routing_key or self.producer_routing_key,
                body=message_body,
                properties=pika.BasicProperties(
                    delivery_mode=2,  # Make message persistent
                    content_type='application/json'
                )
            )
            
            logger.info(f"📤 Published message to {exchange or self.transaction_exchange} -> {routing_key or self.producer_routing_key}")
            logger.debug(f"Message: {message}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to publish message: {e}")
            return False
    
    def set_message_handler(self, handler: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """
        Set the callback function to process incoming messages
        
        Args:
            handler: Function that takes message dict and returns response dict
        """
        self.message_handler = handler
    
    def _on_message(self, ch, method, properties, body):
        """Internal callback for processing messages"""
        try:
            # Parse message
            message = json.loads(body.decode('utf-8'))
            queue_name = method.routing_key
            logger.info(f"📥 Received message from {queue_name}: {message}")
            
            # Process message with handler
            if self.message_handler:
                # Pass queue info to handler
                response = self.message_handler(message, queue_name=queue_name, routing_key=method.routing_key)
                
                print(f"queue_name: {queue_name}, routing_key: {method.routing_key}")
                # Publish response if handler returned something
                if response and queue_name == "transaction.ai.request":
                    self.publish_message(response)
                else:
                    logger.info("ℹ️ No response to publish for this queue")
                    
            # Acknowledge message
            ch.basic_ack(delivery_tag=method.delivery_tag)
            logger.info("✅ Message processed successfully")
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in message: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
    
    def start_consuming(self):
        """Start consuming messages from all consumer queues"""
        if not self.channel or self.channel.is_closed:
            logger.error("❌ Channel not open. Call connect() first.")
            return
        
        try:
            self.channel.basic_qos(prefetch_count=1)
            
            # Consume from all queues
            for queue in self.consumer_queues:
                self.channel.basic_consume(
                    queue=queue,
                    on_message_callback=self._on_message
                )
                logger.info(f"🔄 Started consuming from: {queue}")
            
            # Consume from dynamically added queues
            for queue_name, (routing_key, handler) in self._dynamic_queues.items():
                self.channel.basic_consume(
                    queue=queue_name,
                    on_message_callback=lambda ch, method, properties, body, h=handler:
                        self._on_message_with_handler(ch, method, properties, body, h)
                )
                logger.info(f"🔄 Started consuming from dynamic queue: {queue_name}")
            
            self._consuming = True
            self.channel.start_consuming()
            
        except KeyboardInterrupt:
            logger.info("⏸️ Stopped consuming (KeyboardInterrupt)")
            self.channel.stop_consuming()
            
        except Exception as e:
            logger.error(f"❌ Error consuming messages: {e}")
            self._consuming = False
    
    def start_consuming_background(self):
        """Start consuming in a background thread"""
        if self.consumer_thread and self.consumer_thread.is_alive():
            logger.warning("⚠️ Consumer thread already running")
            return
        
        self.consumer_thread = threading.Thread(target=self.start_consuming, daemon=True)
        self.consumer_thread.start()
        logger.info("🚀 Consumer thread started in background")
    
    def stop_consuming(self):
        """Stop consuming messages"""
        if self.channel and not self.channel.is_closed:
            self.channel.stop_consuming()
            self._consuming = False
            logger.info("⏸️ Stopped consuming messages")
    
    def add_consumer_queue(self, queue_name: str, routing_key: Optional[str] = None, handler: Optional[Callable] = None):
        """
        Dynamically add a new consumer queue
        
        Args:
            queue_name: Name of the queue to consume from
            routing_key: Routing key for binding (defaults to queue_name)
            handler: Optional custom handler for this queue (defaults to global handler)
        """
        routing_key = routing_key or queue_name
        handler = handler or self.message_handler
        
        # Store in registry
        self._dynamic_queues[queue_name] = (routing_key, handler)
        
        # If already connected, declare and bind immediately
        if self.channel and not self.channel.is_closed:
            try:
                self.channel.queue_declare(queue=queue_name, durable=True)
                self.channel.queue_bind(
                    exchange=self.exchange,
                    queue=queue_name,
                    routing_key=routing_key
                )
                
                # If already consuming, start consuming from this queue
                if self._consuming:
                    self.channel.basic_consume(
                        queue=queue_name,
                        on_message_callback=lambda ch, method, properties, body: 
                            self._on_message_with_handler(ch, method, properties, body, handler)
                    )
                    logger.info(f"➕ Added and started consuming from: {queue_name}")
                else:
                    logger.info(f"➕ Added queue: {queue_name} (will consume on start)")
                    
            except Exception as e:
                logger.error(f"❌ Failed to add queue {queue_name}: {e}")
        else:
            logger.info(f"➕ Queued for registration: {queue_name}")
    
    def _on_message_with_handler(self, ch, method, properties, body, handler: Optional[Callable] = None):
        """Internal callback for processing messages with custom handler"""
        try:
            # Parse message
            message = json.loads(body.decode('utf-8'))
            logger.info(f"📥 Received message on {method.routing_key}: {message}")
            
            # Use custom handler or fallback to global
            msg_handler = handler or self.message_handler
            
            # Process message with handler
            if msg_handler:
                response = msg_handler(message)
                
                # Publish response if handler returned something
                if response:
                    self.publish_message(response)
                    
            # Acknowledge message
            ch.basic_ack(delivery_tag=method.delivery_tag)
            logger.info("✅ Message processed successfully")
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in message: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)


# Singleton instance
_rabbitmq_service: Optional[RabbitMQService] = None


def get_rabbitmq_service() -> RabbitMQService:
    """Get or create RabbitMQ service singleton"""
    global _rabbitmq_service
    if _rabbitmq_service is None:
        _rabbitmq_service = RabbitMQService()
    return _rabbitmq_service
