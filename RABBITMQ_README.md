# RabbitMQ Integration for FastAPI Transaction Classifier

This FastAPI service integrates with RabbitMQ to communicate with a Spring Boot microservice.

## Architecture

```
Spring Boot Microservice
         ↓ (publishes)
[category.to.ai.queue] Queue ← FastAPI Consumer
         ↓ (processes)
FastAPI Handler (classify, add_example, etc.)
         ↓ (publishes)
[ai.category.update.queue] Queue → Spring Boot Consumer
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
RABBITMQ_USERNAME=guest
RABBITMQ_PASSWORD=guest
RABBITMQ_VIRTUAL_HOST=/
RABBITMQ_TRANSACTION_EXCHANGE=transaction.exchange
RABBITMQ_AI_EXCHANGE=ai.exchange
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

- **Queue Name**: `category.to.ai.queue`
- **Routing Key**: `category.to.ai`
- **Purpose**: Receives classification requests from Spring Boot

### Producer Queue (Sends to Spring Boot)

- **Queue Name**: `ai.category.update.queue`
- **Routing Key**: `ai.category.update`
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

- `category.to.ai.queue`
- `transaction.to.ai.queue`
- `ai.category.update.queue`
- Exchanges: `transaction.exchange`, `ai.exchange`

### 3. Test Message Flow

**Send test message via RabbitMQ Management UI:**

- Go to Queues → `category.to.ai.queue`
- Publish message:

```json
{
  "action": "classify",
  "userId": "test_user",
  "text": "grab di truong"
}
```

**Check response in `ai.category.update.queue` queue**

### 4. Monitor Logs

```bash
# FastAPI logs will show:
# 📥 Received message: {...}
# ✅ Message processed successfully
# 📤 Published message to ai.category.update
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

rabbitmq.exchange.transaction=transaction.exchange
rabbitmq.exchange.ai=ai.exchange
rabbitmq.queue.request=category.to.ai.queue
rabbitmq.queue.response=ai.category.update.queue
rabbitmq.routing-key.request=category.to.ai
rabbitmq.routing-key.response=ai.category.update
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
          "transaction.exchange",
          "category.to.ai",
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
| RABBITMQ_TRANSACTION_EXCHANGE | transaction.exchange | Transaction exchange  |
| RABBITMQ_AI_EXCHANGE    | ai.exchange              | AI exchange           |
| RABBITMQ_CONSUMER_QUEUE | category.to.ai.queue     | Consumer queue        |
| RABBITMQ_PRODUCER_QUEUE | ai.category.update.queue | Producer queue        |
