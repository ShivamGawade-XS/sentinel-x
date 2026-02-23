"""
API routers package for Sentinel-X.
Imports all route modules for automatic registration.
"""

from routers.threats import router as threats_router
from routers.events import router as events_router
from routers.statistics import router as statistics_router

__all__ = ["threats_router", "events_router", "statistics_router"]
