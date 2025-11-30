#!/usr/bin/env python3
"""
Test RabbitMQ integration
Run this to test message publishing and consuming
"""
import json
import time
from rabbitmq_service import get_rabbitmq_service
from message_handler import handle_message


def test_rabbitmq_connection():
    """Test basic RabbitMQ connection"""
    print("=" * 60)
    print("Testing RabbitMQ Connection")
    print("=" * 60)
    
    rabbitmq = get_rabbitmq_service()
    
    try:
        rabbitmq.connect()
        print("✅ Successfully connected to RabbitMQ")
        print(f"   Host: {rabbitmq.host}:{rabbitmq.port}")
        print(f"   Exchange: {rabbitmq.exchange}")
        print(f"   Consumer Queue: {rabbitmq.consumer_queue}")
        print(f"   Producer Queue: {rabbitmq.producer_queue}")
        return True
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return False
    finally:
        rabbitmq.disconnect()


def test_message_handler():
    """Test message handler logic"""
    print("\n" + "=" * 60)
    print("Testing Message Handler")
    print("=" * 60)
    
    # Test classify action
    test_messages = [
        {
            "action": "classify",
            "userId": "test_user_1",
            "text": "grab di truong"
        },
        {
            "action": "clean_text",
            "text": "mua com 50k"
        },
        {
            "action": "init_user",
            "userId": "test_user_2"
        }
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n--- Test {i} ---")
        print(f"Input: {json.dumps(message, indent=2)}")
        
        response = handle_message(message)
        print(f"Output: {json.dumps(response, indent=2, ensure_ascii=False)}")
        
        if response.get("status") == "success":
            print("✅ Handler processed successfully")
        else:
            print("❌ Handler returned error")


def test_publish_consume():
    """Test publishing and consuming messages"""
    print("\n" + "=" * 60)
    print("Testing Publish/Consume")
    print("=" * 60)
    print("⚠️  This test requires RabbitMQ to be running")
    print("    Start RabbitMQ: docker run -d -p 5672:5672 -p 15672:15672 rabbitmq:3-management")
    print()
    
    rabbitmq = get_rabbitmq_service()
    
    try:
        rabbitmq.connect()
        
        # Test publishing
        test_response = {
            "status": "success",
            "action": "test",
            "message": "This is a test message from Python",
            "timestamp": time.time()
        }
        
        print(f"📤 Publishing test message...")
        success = rabbitmq.publish_message(test_response)
        
        if success:
            print("✅ Message published successfully")
            print(f"   Queue: {rabbitmq.producer_queue}")
            print(f"   Message: {json.dumps(test_response, indent=2)}")
        else:
            print("❌ Failed to publish message")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure RabbitMQ is running:")
        print("   docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3-management")
        
    finally:
        rabbitmq.disconnect()


if __name__ == "__main__":
    print("\n🧪 RabbitMQ Integration Test Suite\n")
    
    # Test 1: Connection
    connection_ok = test_rabbitmq_connection()
    
    # Test 2: Message Handler
    test_message_handler()
    
    # Test 3: Publish/Consume (only if connection works)
    if connection_ok:
        test_publish_consume()
    else:
        print("\n⚠️  Skipping publish/consume test (connection failed)")
        print("   Please start RabbitMQ and try again")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
    print("\n💡 Next steps:")
    print("   1. Start RabbitMQ: docker run -d -p 5672:5672 -p 15672:15672 rabbitmq:3-management")
    print("   2. Start FastAPI: uvicorn main:app --reload")
    print("   3. Check status: curl http://localhost:8000/rabbitmq/status")
    print("   4. View RabbitMQ UI: http://localhost:15672 (guest/guest)")
    print()
