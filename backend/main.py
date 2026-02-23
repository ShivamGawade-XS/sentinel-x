"""
Main entry point for Sentinel-X FastAPI application.
Initializes FastAPI app, middleware, database, and routes.
"""

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZIPMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware

from config import Config
from database import init_db, get_db, health_check
from schemas import HealthCheckResponse, ErrorResponse

# Configure logging
logging.basicConfig(
    level=Config.LOG_LEVEL,
    format=Config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


# ==================== Request/Response Logging Middleware ====================

class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all HTTP requests and responses.
    Records timing, status codes, and error details.
    """
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            logger.info(
                f"{request.method} {request.url.path} - "
                f"Status: {response.status_code} - "
                f"Time: {process_time:.3f}s"
            )
            
            # Add process time header
            response.headers["X-Process-Time"] = str(process_time)
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"{request.method} {request.url.path} - "
                f"Error: {str(e)} - Time: {process_time:.3f}s",
                exc_info=True
            )
            raise


# ==================== Lifespan Events ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle startup and shutdown events for the FastAPI application.
    """
    # Startup
    logger.info("Starting Sentinel-X application...")
    try:
        init_db()
        logger.info("Database initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Sentinel-X application...")


# ==================== FastAPI App Initialization ====================

app = FastAPI(
    title="Sentinel-X Threat Detection API",
    description="Advanced threat detection and analysis platform",
    version=Config.APP_VERSION,
    docs_url=Config.API_DOCS_URL,
    openapi_url=Config.API_OPENAPI_URL,
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(GZIPMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Global Exception Handlers ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.warning(
        f"HTTP Exception: {exc.status_code} - {exc.detail} "
        f"[{request.method} {request.url.path}]"
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            status="error",
            code=exc.status_code,
            message=exc.detail,
            timestamp=datetime.utcnow()
        ).model_dump()
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            status="error",
            code=422,
            message="Validation failed",
            details={"errors": exc.errors()},
            timestamp=datetime.utcnow()
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(
        f"Unexpected error: {str(exc)} [{request.method} {request.url.path}]",
        exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            status="error",
            code=500,
            message="Internal server error",
            timestamp=datetime.utcnow()
        ).model_dump()
    )


# ==================== Health Check Endpoints ====================

@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check_endpoint():
    """
    Health check endpoint that verifies API and database connectivity.
    """
    db_status = health_check()
    return HealthCheckResponse(
        status="healthy" if db_status else "degraded",
        timestamp=datetime.utcnow(),
        database=db_status,
        version=Config.APP_VERSION
    )


@app.get("/api/v1/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check_v1():
    """
    API v1 health check endpoint.
    """
    return await health_check_endpoint()


# ==================== API Routes ====================

# Import and register route modules
from routers import threats_router, events_router, statistics_router

app.include_router(threats_router)
app.include_router(events_router)
app.include_router(statistics_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
