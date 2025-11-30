# RabbitMQ Integration for FastAPI Transaction Classifier

This FastAPI service integrates with RabbitMQ to communicate with a Spring Boot microservice.

## Architecture

```
Spring Boot Microservice
         ↓ (publishes)
[python.classify.request] Queue ← FastAPI Consumer
         ↓ (processes)
FastAPI Handler (classify, add_example, etc.)
         ↓ (publishes)
[python.classify.response] Queue → Spring Boot Consumer
```

## Setup

### 1. Install RabbitMQ

**Using Docker:**

```bash
docker run -d --name rabbitmq \
  -p 5672:5672 \
  -p 15672:15672 \
  rabbitmq:3-management
```

**Or install locally:**

- macOS: `brew install rabbitmq`
- Ubuntu: `sudo apt-get install rabbitmq-server`

### 2. Configure Environment

Copy `.env.example` to `.env` and update values:

```bash
cp .env.example .env
```

Edit `.env`:

```env
RABBITMQ_HOST=localhost
RABBITMQ_PORT=5672
RABBITMQ_USER=guest
RABBITMQ_PASSWORD=guest
```

### 3. Install Dependencies

```bash
pip install pika python-dotenv
```

### 4. Start FastAPI Server

```bash
uvicorn main:app --reload
```

## Queue Configuration

### Consumer Queue (Receives from Spring Boot)

- **Queue Name**: `category.ai.queue`
- **Routing Key**: `category.ai.request`
- **Purpose**: Receives classification requests from Spring Boot

### Producer Queue (Sends to Spring Boot)

- **Queue Name**: `python.classify.response`
- **Routing Key**: `transaction.classify.response`
- **Purpose**: Sends classification results back to Spring Boot

## Message Format

### Request (from Spring Boot → FastAPI)

```json
{
  "action": "classify",
  "userId": "user123",
  "text": "grab di truong"
}
```

**Supported Actions:**

- `classify` - Classify a transaction
- `add_example` - Add training example
- `add_category` - Add custom category
- `init_user` - Initialize user categories
- `clean_text` - Clean and parse transaction text

### Response (from FastAPI → Spring Boot)

```json
{
  "status": "success",
  "action": "classify",
  "userId": "user123",
  "data": {
    "categoryId": "3",
    "categoryName": "Transport",
    "confidence": 0.89
  },
  "message": "Classification completed successfully"
}
```

## API Endpoints

### Health Check

```bash
GET http://localhost:8000/rabbitmq/status
```

### Manual Publish (Testing)

```bash
POST http://localhost:8000/rabbitmq/publish
Content-Type: application/json

{
  "status": "success",
  "action": "test",
  "message": "Test message"
}
```

## Testing

### 1. Check RabbitMQ Management UI

Visit: http://localhost:15672 (guest/guest)

### 2. Verify Queues Created

- `category.ai.queue`
- `python.classify.response`
- Exchange: `transaction_exchange`

### 3. Test Message Flow

**Send test message via RabbitMQ Management UI:**

- Go to Queues → `category.ai.queue`
- Publish message:

```json
{
  "action": "classify",
  "userId": "test_user",
  "text": "grab di truong"
}
```

**Check response in `python.classify.response` queue**

### 4. Monitor Logs

```bash
# FastAPI logs will show:
# 📥 Received message: {...}
# ✅ Message processed successfully
# 📤 Published message to transaction.classify.response
```

## Spring Boot Integration

### Maven Dependency

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-amqp</artifactId>
</dependency>
```

### Application Properties

```properties
spring.rabbitmq.host=localhost
spring.rabbitmq.port=5672
spring.rabbitmq.username=guest
spring.rabbitmq.password=guest

rabbitmq.exchange.name=transaction_exchange
rabbitmq.queue.request=category.ai.queue
rabbitmq.queue.response=python.classify.response
rabbitmq.routing-key.request=category.ai.request
rabbitmq.routing-key.response=transaction.classify.response
```

### Publisher (Spring Boot sends to Python)

```java
@Service
public class TransactionPublisher {
    @Autowired
    private RabbitTemplate rabbitTemplate;

    public void classifyTransaction(String userId, String text) {
        Map<String, Object> message = Map.of(
            "action", "classify",
            "userId", userId,
            "text", text
        );

        rabbitTemplate.convertAndSend(
            "transaction_exchange",
            "category.ai.request",
            message
        );
    }
}
```

### Consumer (Spring Boot receives from Python)

```java
@Service
public class TransactionConsumer {

    @RabbitListener(queues = "${rabbitmq.queue.response}")
    public void handleResponse(Map<String, Object> response) {
        String status = (String) response.get("status");
        String action = (String) response.get("action");
        Object data = response.get("data");

        // Process response...
    }
}
```

## Troubleshooting

### RabbitMQ Connection Failed

- Check if RabbitMQ is running: `sudo systemctl status rabbitmq-server`
- Verify credentials in `.env`
- Check port 5672 is accessible

### Messages Not Being Consumed

- Check FastAPI logs for "Started consuming"
- Verify queue bindings in RabbitMQ Management UI
- Check message format matches expected schema

### No Response Received

- Check FastAPI processing logs
- Verify producer queue exists
- Check Spring Boot consumer is listening

## Environment Variables Reference

| Variable                | Default                  | Description           |
| ----------------------- | ------------------------ | --------------------- |
| RABBITMQ_HOST           | localhost                | RabbitMQ server host  |
| RABBITMQ_PORT           | 5672                     | RabbitMQ server port  |
| RABBITMQ_USER           | guest                    | RabbitMQ username     |
| RABBITMQ_PASSWORD       | guest                    | RabbitMQ password     |
| RABBITMQ_VHOST          | /                        | RabbitMQ virtual host |
| RABBITMQ_EXCHANGE       | transaction_exchange     | Exchange name         |
| RABBITMQ_CONSUMER_QUEUE | category.ai.queue        | Consumer queue        |
| RABBITMQ_PRODUCER_QUEUE | python.classify.response | Producer queue        |
