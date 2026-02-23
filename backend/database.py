"""
Database configuration and session management for Sentinel-X.
Handles SQLAlchemy engine creation, session management, and database utilities.
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from typing import Generator
import logging

from config import Config
from models import Base

logger = logging.getLogger(__name__)

# Create database engine
engine = create_engine(
    Config.DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # Verify connections before using
    echo=Config.SQLALCHEMY_ECHO,
)

# Create session factory
SessionLocal = sessionmaker(
    bind=engine,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI to get database session.
    
    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initialize database by creating all tables.
    Should be called once at application startup.
    """
    try:
        logger.info("Initializing database...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}", exc_info=True)
        raise


def drop_all():
    """
    Drop all tables from database.
    WARNING: This will delete all data!
    Only use this for testing or cleanup.
    """
    try:
        logger.warning("Dropping all database tables!")
        Base.metadata.drop_all(bind=engine)
        logger.info("All tables dropped successfully!")
    except Exception as e:
        logger.error(f"Failed to drop tables: {str(e)}", exc_info=True)
        raise


def reset_db():
    """
    Reset database by dropping and recreating all tables.
    WARNING: This will delete all data!
    """
    try:
        logger.warning("Resetting database!")
        drop_all()
        init_db()
        logger.info("Database reset successfully!")
    except Exception as e:
        logger.error(f"Failed to reset database: {str(e)}", exc_info=True)
        raise


def health_check() -> bool:
    """
    Check database connectivity.
    
    Returns:
        bool: True if database is accessible, False otherwise
    """
    try:
        with engine.connect() as connection:
            connection.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        return False
