"""
Statistics and reporting API endpoints.
Provides threat analysis, reports generation, and system statistics.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func

from database import get_db
from models import (
    Threat, ThreatAnalysis, SecurityEvent, ThreatLevelEnum, ThreatTypeEnum
)
from schemas import ThreatLevel, ThreatType, StatisticsResponse

router = APIRouter(
    prefix="/api/v1/statistics",
    tags=["Statistics"],
)

logger = logging.getLogger(__name__)


# ==================== Threat Statistics ====================

@router.get(
    "/threats",
    response_model=StatisticsResponse,
    summary="Get threat detection statistics"
)
async def get_threat_statistics(
    days: int = Query(30, ge=1, le=365, description="Look back period in days"),
    db: Session = Depends(get_db)
) -> StatisticsResponse:
    """
    Get comprehensive threat detection statistics.
    
    Includes:
    - Total threats detected
    - Threats by severity level
    - Threats by type
    - Detection metrics and trends
    
    **Query Parameters:**
    - **days**: Analysis period in days (default: 30)
    """
    try:
        logger.info(f"Fetching threat statistics for the last {days} days")
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        today_cutoff = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Total threats
        total_threats = db.query(func.count(Threat.id)).filter(
            Threat.detected_at >= cutoff_date
        ).scalar() or 0
        
        # Threats today
        threats_today = db.query(func.count(Threat.id)).filter(
            Threat.detected_at >= today_cutoff
        ).scalar() or 0
        
        # Threats this week
        week_cutoff = datetime.utcnow() - timedelta(days=7)
        threats_this_week = db.query(func.count(Threat.id)).filter(
            Threat.detected_at >= week_cutoff
        ).scalar() or 0
        
        # Count by severity
        severity_counts = db.query(
            Threat.threat_level,
            func.count(Threat.id).label("count")
        ).filter(
            Threat.detected_at >= cutoff_date
        ).group_by(
            Threat.threat_level
        ).all()
        
        severity_dict = {level.value: 0 for level in ThreatLevelEnum}
        for level, count in severity_counts:
            severity_dict[level.value] = count
        
        critical_threats = severity_dict.get("CRITICAL", 0)
        high_severity = severity_dict.get("HIGH", 0)
        medium_severity = severity_dict.get("MEDIUM", 0)
        low_severity = severity_dict.get("LOW", 0)
        
        # Count by threat type
        type_counts = db.query(
            Threat.threat_type,
            func.count(Threat.id).label("count")
        ).filter(
            Threat.detected_at >= cutoff_date
        ).group_by(
            Threat.threat_type
        ).all()
        
        top_threat_types = {threat_type.value: count for threat_type, count in type_counts}
        
        # Calculate average detection time (mock for now)
        avg_detection_time_ms = 145.3
        detection_rate_percentage = 96.5
        
        logger.info(f"Statistics calculated - Total threats: {total_threats}")
        
        return StatisticsResponse(
            total_threats=total_threats,
            threats_today=threats_today,
            threats_this_week=threats_this_week,
            critical_threats=critical_threats,
            high_severity=high_severity,
            medium_severity=medium_severity,
            low_severity=low_severity,
            top_threat_types=top_threat_types,
            avg_detection_time_ms=avg_detection_time_ms,
            detection_rate_percentage=detection_rate_percentage
        )
        
    except Exception as e:
        logger.error(f"Error fetching statistics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch statistics")


@router.get(
    "/threats/by-type",
    summary="Get threat distribution by type"
)
async def get_threats_by_type(
    days: int = Query(30, ge=1, le=365, description="Look back period in days"),
    limit: int = Query(10, ge=1, le=50, description="Top N threat types to return"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get threat distribution by type.
    
    Returns the most common threat types detected in the specified period.
    """
    try:
        logger.info(f"Fetching threat distribution by type")
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        type_counts = db.query(
            Threat.threat_type,
            func.count(Threat.id).label("count")
        ).filter(
            Threat.detected_at >= cutoff_date
        ).group_by(
            Threat.threat_type
        ).order_by(
            func.count(Threat.id).desc()
        ).limit(limit).all()
        
        result = {threat_type.value: count for threat_type, count in type_counts}
        
        return {
            "period_days": days,
            "total_types": len(result),
            "by_type": result,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching threat types: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch threat type statistics")


@router.get(
    "/threats/by-severity",
    summary="Get threat distribution by severity"
)
async def get_threats_by_severity(
    days: int = Query(30, ge=1, le=365, description="Look back period in days"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get threat distribution by severity level.
    """
    try:
        logger.info(f"Fetching threat distribution by severity")
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        severity_counts = db.query(
            Threat.threat_level,
            func.count(Threat.id).label("count")
        ).filter(
            Threat.detected_at >= cutoff_date
        ).group_by(
            Threat.threat_level
        ).all()
        
        result = {level.value: count for level, count in severity_counts}
        
        return {
            "period_days": days,
            "by_severity": result,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching severity statistics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch severity statistics")


# ==================== System Statistics ====================

@router.get(
    "/system/overview",
    summary="Get system overview statistics"
)
async def get_system_overview(
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get general system statistics and health metrics.
    """
    try:
        logger.info("Fetching system overview statistics")
        
        # Count all records
        total_threats = db.query(func.count(Threat.id)).scalar() or 0
        total_analyses = db.query(func.count(ThreatAnalysis.id)).scalar() or 0
        total_events = db.query(func.count(SecurityEvent.id)).scalar() or 0
        
        # Statistics from last 24 hours
        cutoff_24h = datetime.utcnow() - timedelta(hours=24)
        threats_24h = db.query(func.count(Threat.id)).filter(
            Threat.detected_at >= cutoff_24h
        ).scalar() or 0
        
        events_24h = db.query(func.count(SecurityEvent.id)).filter(
            SecurityEvent.created_at >= cutoff_24h
        ).scalar() or 0
        
        # Get critical threats
        critical_threats = db.query(func.count(Threat.id)).filter(
            Threat.threat_level == ThreatLevelEnum.CRITICAL
        ).scalar() or 0
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "all_time": {
                "total_threats": total_threats,
                "total_analyses": total_analyses,
                "total_events": total_events
            },
            "last_24_hours": {
                "threats": threats_24h,
                "events": events_24h
            },
            "active_alerts": {
                "critical": critical_threats
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching system overview: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch system overview")


# ==================== Trend Analysis ====================

@router.get(
    "/trends/threat-detection",
    summary="Get threat detection trends"
)
async def get_threat_detection_trends(
    days: int = Query(30, ge=1, le=365, description="Look back period in days"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get threat detection trends over time.
    
    Returns daily threat detection counts for trend analysis.
    """
    try:
        logger.info(f"Fetching threat detection trends for {days} days")
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Query threats with date grouping
        from sqlalchemy import cast, Date
        trends = db.query(
            cast(Threat.detected_at, Date).label("date"),
            func.count(Threat.id).label("count")
        ).filter(
            Threat.detected_at >= cutoff_date
        ).group_by(
            cast(Threat.detected_at, Date)
        ).order_by(
            cast(Threat.detected_at, Date)
        ).all()
        
        trend_data = {str(date): count for date, count in trends}
        
        return {
            "period_days": days,
            "generated_at": datetime.utcnow().isoformat(),
            "daily_threats": trend_data,
            "total_threats": sum(trend_data.values())
        }
        
    except Exception as e:
        logger.error(f"Error fetching trends: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch trend data")
