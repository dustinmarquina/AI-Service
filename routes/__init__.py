"""API Routes"""

from .categories import router as categories_router
from .analysis import router as analysis_router
from .chat import router as chat_router
from .rabbitmq import router as rabbitmq_router

__all__ = [
    "categories_router",
    "analysis_router",
    "chat_router",
    "rabbitmq_router",
]
