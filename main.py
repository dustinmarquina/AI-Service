"""
AI Service - FastAPI Application

A comprehensive FastAPI service for transaction categorization, financial predictions,
AI chat, and RabbitMQ message processing.

Author: AI Service Team
Version: 2.0.0
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Defer heavy imports until startup to avoid preloading during `uvicorn --reload`.
# They will be imported inside the `lifespan` context manager below.

# ============================================
# Logging Configuration
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================
# Application Lifecycle Management
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle events.
    
    Handles:
    - RabbitMQ connection initialization
    - Background consumer startup
    - Graceful shutdown and cleanup
    """
    from rabbitmq_service import get_rabbitmq_service
    from message_handler import handle_message
    from agents.orchestrator.graph import build_main_graph
    from routes import (
        auth_router,
        categories_router,
        chat_router,
        rabbitmq_router,
    )

    # Re-register routers at startup (safe to call multiple times)
    app.include_router(auth_router)
    app.include_router(categories_router)
    app.include_router(chat_router)
    app.include_router(rabbitmq_router)
    # Initialize RabbitMQ service instance after imports
    rabbitmq = get_rabbitmq_service()
    try:
        # Startup
        # Initialize the orchestrator graph used by chat and routing
        app.state.main_graph = await build_main_graph()
        logger.info("✅ Main graph initialized")
        logger.info("🚀 Starting AI Service...")
        rabbitmq.connect()
        rabbitmq.set_message_handler(handle_message)
        rabbitmq.start_consuming_background()
        logger.info("✅ RabbitMQ consumer started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to start RabbitMQ: {e}")
        logger.info("⚠️  Continuing without RabbitMQ...")
        yield
        
    finally:
        # Shutdown
        logger.info("🛑 Shutting down AI Service...")
        try:
            rabbitmq.stop_consuming()
            rabbitmq.disconnect()
            logger.info("✅ Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# ============================================
# FastAPI Application
# ============================================

app = FastAPI(
    title="AI Service API",
    description="AI-powered financial analysis, transaction categorization, and chat service",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ============================================
# Middleware Configuration
# ============================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Exception Handlers
# ============================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "An internal server error occurred",
            "detail": str(exc) if app.debug else None,
        },
    )


# Routers are included at startup inside the lifespan context to avoid heavy imports on module load.


# ============================================
# Health Check & Root Endpoints
# ============================================

@app.get("/", tags=["Health"])
def root():
    """Root endpoint - API status check"""
    return {
        "service": "AI Service",
        "status": "running",
        "version": "2.0.0",
        "message": "Welcome to AI Service API",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    
    Returns service status and component health.
    """
    from rabbitmq_service import get_rabbitmq_service
    rabbitmq = get_rabbitmq_service()
    rabbitmq_healthy = rabbitmq.connection and rabbitmq.connection.is_open
    
    return {
        "status": "healthy",
        "service": "ai-service",
        "components": {
            "api": "ok",
            "rabbitmq": "ok" if rabbitmq_healthy else "degraded",
        },
    }
