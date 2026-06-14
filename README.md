# AI Service

Backend service for a Vietnamese personal finance assistant. The project combines:

- FastAPI for HTTP endpoints and SSE-style chat responses
- LangGraph-based orchestration for planning, SQL queries, and tool execution
- RabbitMQ for transaction/category event processing
- MongoDB-backed transaction classification and per-user category memory
- Ollama/LLM utilities for seeded example generation and chat features

## What It Does

This service is built around two main workflows:

1. Conversational finance assistant
   - receives chat requests
   - routes them through planner / executor / SQL-agent paths
   - handles tool calls, user identity resolution, and interrupt/resume flows

2. Event-driven transaction intelligence
   - consumes category and transaction events from RabbitMQ
   - classifies transaction descriptions into categories
   - updates per-user category examples and embedding centroids

## Main Components

- [main.py](./main.py): FastAPI app entrypoint and lifespan startup/shutdown
- [routes/](./routes): HTTP endpoints, including chat streaming
- [agents/orchestrator/graph/](./agents/orchestrator/graph): planner, executor, SQL agent, and graph wiring
- [rabbitmq_service.py](./rabbitmq_service.py): queue connection, consume/publish logic
- [message_handler.py](./message_handler.py): transaction/category event handlers
- [transaction_classifying.py](./transaction_classifying.py): embedding-based transaction categorization
- [llm_client.py](./llm_client.py): local/cloud Ollama wrapper for prompt-based generation

## Tech Stack

- Python
- FastAPI
- LangGraph
- PostgreSQL
- MongoDB
- RabbitMQ
- Sentence Transformers
- Ollama

## Getting Started

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

Create `.env` and set the services you use. Typical values include:

```env
POSTGRES_URI=
TRANSACTION_API_URL=
CATEGORY_API_URL=
TRANSACTION_USER_ID=

RABBITMQ_HOST=
RABBITMQ_PORT=
RABBITMQ_USERNAME=
RABBITMQ_PASSWORD=
RABBITMQ_VIRTUAL_HOST=

groq_api_key=
OLLAMA_API_KEY=
OLLAMA_LOCAL_HOST=
```

### 3. Run the API

```bash
uvicorn main:app --reload
```

Open:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`

## Demo

Main graph from Lang Smith

<img width="894" height="922" alt="image" src="https://github.com/user-attachments/assets/d0b61207-9b89-487e-b977-f9426026a558" />


<img width="323" height="698" alt="image" src="https://github.com/user-attachments/assets/fb52252e-6d60-4cd7-b45c-37e32e2c5bab" />

<img width="293" height="401" alt="att Rr_46eqw41-BwtqoPVXUW9TXZEvbAg-xKJ1a1FuBkoI" src="https://github.com/user-attachments/assets/d3820612-6295-448d-b12d-d3c6ad4a261f" />



Suggested layout:

```md
## Demo

### Chat Flow
![Chat Flow](./assets/demo/chat-flow.png)

### Wallet Selection Interrupt
![Wallet Selection](./assets/demo/wallet-selection.png)

### Transaction Classification
![Transaction Classification](./assets/demo/transaction-classification.png)
```

If you want, create an `assets/demo/` folder and place your images there so the links stay simple.

## Notes

- This project is still in active iteration and some paths are prototype-level.
- The most interesting parts are orchestration, event-driven classification, and service integration rather than polished product UX.

## Development

Run the test suite with:

```bash
venv/bin/python -m unittest
```
