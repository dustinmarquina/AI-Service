"""API Routes"""

from .auth import router as auth_router
from .categories import router as categories_router
from .chat import router as chat_router
from .rabbitmq import router as rabbitmq_router

__all__ = [
    "auth_router",
    "categories_router",
    "chat_router",
    "rabbitmq_router",
]
