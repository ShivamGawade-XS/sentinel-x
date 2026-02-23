"""
Security events API endpoints.
Handles logging and retrieval of security events.
"""

import logging
from typing import List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from sqlalchemy.orm import Session
from uuid import uuid4

from database import get_db
from models import SecurityEvent as SecurityEventModel, ThreatLevelEnum
from schemas import SecurityEventRequest, SecurityEventResponse, ThreatLevel

router = APIRouter(
    prefix="/api/v1/events",
    tags=["Security Events"],
)

logger = logging.getLogger(__name__)


# ==================== Event Logging ====================

@router.post(
    "/log",
    response_model=SecurityEventResponse,
    summary="Log security event",
    status_code=201
)
async def log_security_event(
    event: SecurityEventRequest = Body(..., example={
        "event_type": "SUSPICIOUS_LOGIN",
        "severity": "HIGH",
        "source": "192.168.1.100",
        "description": "Multiple failed login attempts detected",
        "metadata": {"attempts": 5, "user": "admin"}
    }),
    db: Session = Depends(get_db)
) -> SecurityEventResponse:
    """
    Log a security event for threat correlation and analysis.
    
    Automatically processed and stored for future reference.
    """
    try:
        logger.info(f"Logging security event: {event.event_type} from {event.source}")
        
        event_id = f"EVT-{uuid4().hex[:12].upper()}"
        
        # Create and save event
        db_event = SecurityEventModel(
            event_id=event_id,
            event_type=event.event_type,
            severity=ThreatLevelEnum[event.severity.value],
            source=event.source,
            description=event.description,
            metadata=event.metadata or {}
        )
        
        db.add(db_event)
        db.commit()
        db.refresh(db_event)
        
        logger.info(f"Event logged successfully: {event_id}")
        
        return SecurityEventResponse(
            event_id=db_event.event_id,
            event_type=db_event.event_type,
            severity=ThreatLevel[db_event.severity.value],
            source=db_event.source,
            description=db_event.description,
            timestamp=db_event.created_at,
            metadata=db_event.metadata
        )
        
    except Exception as e:
        logger.error(f"Error logging security event: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to log event")


# ==================== Event Retrieval ====================

@router.get(
    "",
    response_model=List[SecurityEventResponse],
    summary="Retrieve security events with filtering"
)
async def get_security_events(
    limit: int = Query(20, ge=1, le=100, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    severity: Optional[str] = Query(None, description="Filter by severity level"),
    source: Optional[str] = Query(None, description="Filter by event source"),
    days: int = Query(7, ge=1, le=365, description="Look back period in days"),
    db: Session = Depends(get_db)
) -> List[SecurityEventResponse]:
    """
    Retrieve recent security events with optional filtering.
    
    **Query Parameters:**
    - **limit**: Maximum number of results (1-100)
    - **offset**: Offset for pagination
    - **event_type**: Filter by event type (SUSPICIOUS_LOGIN, NETWORK_ANOMALY, etc.)
    - **severity**: Filter by severity (CRITICAL, HIGH, MEDIUM, LOW, INFO)
    - **source**: Filter by event source (IP, hostname, etc.)
    - **days**: Look back period in days (1-365)
    """
    try:
        logger.info(f"Fetching security events - limit: {limit}, offset: {offset}")
        
        # Build query
        query = db.query(SecurityEventModel)
        
        # Filter by date
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        query = query.filter(SecurityEventModel.created_at >= cutoff_date)
        
        # Filter by event type
        if event_type:
            query = query.filter(SecurityEventModel.event_type.ilike(f"%{event_type}%"))
        
        # Filter by severity
        if severity:
            try:
                query = query.filter(SecurityEventModel.severity == ThreatLevelEnum[severity])
            except KeyError:
                raise HTTPException(status_code=400, detail=f"Invalid severity level: {severity}")
        
        # Filter by source
        if source:
            query = query.filter(SecurityEventModel.source.ilike(f"%{source}%"))
        
        # Apply pagination and sorting
        events = query.order_by(SecurityEventModel.created_at.desc()).offset(offset).limit(limit).all()
        
        # Convert to response models
        responses = [
            SecurityEventResponse(
                event_id=event.event_id,
                event_type=event.event_type,
                severity=ThreatLevel[event.severity.value],
                source=event.source,
                description=event.description,
                timestamp=event.created_at,
                metadata=event.metadata
            )
            for event in events
        ]
        
        logger.info(f"Retrieved {len(events)} events")
        return responses
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving events: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve events")


@router.get(
    "/{event_id}",
    response_model=SecurityEventResponse,
    summary="Get event details by ID"
)
async def get_event_details(
    event_id: str = Query(..., min_length=1, description="Event ID"),
    db: Session = Depends(get_db)
) -> SecurityEventResponse:
    """
    Retrieve detailed information about a specific security event.
    """
    try:
        logger.info(f"Fetching event details: {event_id}")
        
        event = db.query(SecurityEventModel).filter(
            SecurityEventModel.event_id == event_id
        ).first()
        
        if not event:
            raise HTTPException(status_code=404, detail=f"Event not found: {event_id}")
        
        return SecurityEventResponse(
            event_id=event.event_id,
            event_type=event.event_type,
            severity=ThreatLevel[event.severity.value],
            source=event.source,
            description=event.description,
            timestamp=event.created_at,
            metadata=event.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving event details: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve event details")


# ==================== Event Statistics ====================

@router.get(
    "/stats/by-type",
    summary="Get event statistics by type"
)
async def get_events_by_type(
    days: int = Query(7, ge=1, le=365, description="Look back period in days"),
    db: Session = Depends(get_db)
):
    """
    Get event distribution by type over the specified period.
    """
    try:
        from sqlalchemy import func
        
        logger.info(f"Fetching event statistics by type")
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        stats = db.query(
            SecurityEventModel.event_type,
            func.count(SecurityEventModel.id).label("count")
        ).filter(
            SecurityEventModel.created_at >= cutoff_date
        ).group_by(
            SecurityEventModel.event_type
        ).all()
        
        result = {event_type: count for event_type, count in stats}
        
        return {
            "period_days": days,
            "total_events": sum(result.values()),
            "by_type": result
        }
        
    except Exception as e:
        logger.error(f"Error fetching event statistics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")


@router.get(
    "/stats/by-severity",
    summary="Get event statistics by severity"
)
async def get_events_by_severity(
    days: int = Query(7, ge=1, le=365, description="Look back period in days"),
    db: Session = Depends(get_db)
):
    """
    Get event distribution by severity level over the specified period.
    """
    try:
        from sqlalchemy import func
        
        logger.info(f"Fetching event statistics by severity")
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        stats = db.query(
            SecurityEventModel.severity,
            func.count(SecurityEventModel.id).label("count")
        ).filter(
            SecurityEventModel.created_at >= cutoff_date
        ).group_by(
            SecurityEventModel.severity
        ).all()
        
        result = {severity.value: count for severity, count in stats}
        
        return {
            "period_days": days,
            "total_events": sum(result.values()),
            "by_severity": result
        }
        
    except Exception as e:
        logger.error(f"Error fetching severity statistics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")
