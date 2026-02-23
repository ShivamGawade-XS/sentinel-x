"""
Threat detection and analysis API endpoints.
Handles threat detection, analysis, and retrieval operations.
"""

import logging
from typing import List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from sqlalchemy.orm import Session
from uuid import uuid4

from database import get_db
from models import Threat, ThreatIndicator, ThreatAnalysis, ThreatTypeEnum, ThreatLevelEnum, AnalysisStatusEnum
from schemas import (
    ThreatDetectionRequest,
    ThreatDetectionResponse,
    ThreatAnalysisResponse,
    AnalysisStatus,
    ThreatLevel,
    ThreatType,
)

router = APIRouter(
    prefix="/api/v1/threats",
    tags=["Threat Detection"],
)

logger = logging.getLogger(__name__)


# ==================== Threat Detection ====================

@router.post(
    "/detect",
    response_model=ThreatDetectionResponse,
    summary="Detect threats in provided artifacts",
    responses={
        200: {"description": "Threat detected"},
        404: {"description": "No threat detected"},
        400: {"description": "Invalid request parameters"},
        500: {"description": "Internal server error"}
    }
)
async def detect_threat(
    payload: ThreatDetectionRequest = Body(..., example={
        "sample": "5d41402abc4b2a76b9719d911017c592",
        "sample_type": "hash",
        "priority": "HIGH"
    }),
    db: Session = Depends(get_db)
) -> ThreatDetectionResponse:
    """
    Analyze provided artifacts for threats.
    
    Supported sample types:
    - **hash**: File hash (MD5, SHA1, SHA256)
    - **url**: Website or endpoint URL
    - **ip**: IPv4 or IPv6 address
    - **file**: Base64 encoded file content
    - **domain**: Domain name
    - **email**: Email address
    
    Returns detailed threat information if detected.
    """
    try:
        logger.info(f"Processing threat detection for sample: {payload.sample[:20]}...")
        
        # Mock threat detection logic (replace with actual ML model later)
        threat_detected = _analyze_sample(payload.sample, payload.sample_type)
        
        if threat_detected:
            # Create threat record in database
            threat = Threat(
                threat_id=f"THR-{uuid4().hex[:12].upper()}",
                threat_type=ThreatTypeEnum[threat_detected["type"].value],
                threat_level=ThreatLevelEnum[threat_detected["level"].value],
                description=threat_detected["description"],
                confidence=threat_detected.get("confidence", 0.95),
                detected_at=datetime.utcnow()
            )
            
            db.add(threat)
            db.flush()  # Get the threat ID without committing
            
            # Add indicators
            for indicator in threat_detected.get("indicators", []):
                threat_indicator = ThreatIndicator(
                    threat_id=threat.id,
                    name=indicator.name,
                    value=indicator.value,
                    indicator_type=indicator.indicator_type.value,
                    confidence=indicator.confidence
                )
                db.add(threat_indicator)
            
            db.commit()
            logger.info(f"Threat detected and saved: {threat.threat_id}")
            
            # Format response
            response = ThreatDetectionResponse(
                threat_id=threat.threat_id,
                threat_type=ThreatType[threat.threat_type.value],
                threat_level=ThreatLevel[threat.threat_level.value],
                detected_at=threat.detected_at,
                description=threat.description,
                indicators=[
                    {
                        "name": ind.name,
                        "value": ind.value,
                        "indicator_type": ind.indicator_type,
                        "confidence": ind.confidence
                    } for ind in threat.indicators
                ],
                affected_systems=threat_detected.get("systems", []),
                recommendations=threat_detected.get("recommendations", []),
                confidence=threat.confidence
            )
            return response
        else:
            raise HTTPException(
                status_code=404,
                detail="No threat detected for the provided sample"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in threat detection: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ==================== Threat Analysis ====================

@router.post(
    "/analyze",
    response_model=ThreatAnalysisResponse,
    summary="Perform detailed threat analysis"
)
async def analyze_threat(
    payload: ThreatDetectionRequest = Body(...),
    db: Session = Depends(get_db)
) -> ThreatAnalysisResponse:
    """
    Perform comprehensive threat analysis on artifacts.
    
    Includes:
    - Threat identification
    - Indicator extraction
    - Behavioral analysis
    - Detailed threat classification
    """
    try:
        logger.info(f"Starting detailed analysis for: {payload.sample[:20]}...")
        
        analysis_id = f"ANL-{uuid4().hex[:12].upper()}"
        threat_detected = _analyze_sample(payload.sample, payload.sample_type)
        
        # Create analysis record
        analysis = ThreatAnalysis(
            analysis_id=analysis_id,
            sample=payload.sample,
            sample_type=payload.sample_type,
            status=AnalysisStatusEnum.COMPLETED,
            threat_detected=threat_detected is not None,
            processing_time_ms=150.5
        )
        
        if threat_detected:
            # Create threat record
            threat = Threat(
                threat_id=f"THR-{uuid4().hex[:12].upper()}",
                threat_type=ThreatTypeEnum[threat_detected["type"].value],
                threat_level=ThreatLevelEnum[threat_detected["level"].value],
                description=threat_detected["description"],
                confidence=threat_detected.get("confidence", 0.95),
                detected_at=datetime.utcnow()
            )
            
            db.add(threat)
            db.flush()
            
            # Add indicators
            for indicator in threat_detected.get("indicators", []):
                threat_indicator = ThreatIndicator(
                    threat_id=threat.id,
                    name=indicator.name,
                    value=indicator.value,
                    indicator_type=indicator.indicator_type.value,
                    confidence=indicator.confidence
                )
                db.add(threat_indicator)
            
            analysis.threat_id = threat.id
            analysis.analysis_details = threat_detected.get("details", {})
            
            db.add(analysis)
            db.commit()
            
            logger.info(f"Analysis completed: {analysis_id} - Threat: {threat.threat_id}")
            
            threat_response = ThreatDetectionResponse(
                threat_id=threat.threat_id,
                threat_type=ThreatType[threat.threat_type.value],
                threat_level=ThreatLevel[threat.threat_level.value],
                detected_at=threat.detected_at,
                description=threat.description,
                indicators=[
                    {
                        "name": ind.name,
                        "value": ind.value,
                        "indicator_type": ind.indicator_type,
                        "confidence": ind.confidence
                    } for ind in threat.indicators
                ],
                affected_systems=threat_detected.get("systems", []),
                recommendations=threat_detected.get("recommendations", []),
                confidence=threat.confidence
            )
        else:
            db.add(analysis)
            db.commit()
            threat_response = None
            logger.info(f"Analysis completed: {analysis_id} - No threat detected")
        
        return ThreatAnalysisResponse(
            analysis_id=analysis_id,
            status=AnalysisStatus.COMPLETED,
            sample=payload.sample,
            threat_detected=threat_detected is not None,
            threat=threat_response,
            indicators=[],
            analysis_details=threat_detected.get("details", {}) if threat_detected else {},
            processing_time_ms=150.5
        )
        
    except Exception as e:
        logger.error(f"Error in threat analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ==================== Threat Retrieval ====================

@router.get(
    "",
    response_model=List[ThreatDetectionResponse],
    summary="Retrieve detected threats with filtering"
)
async def get_threats(
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    threat_level: Optional[str] = Query(None, description="Filter by threat level"),
    threat_type: Optional[str] = Query(None, description="Filter by threat type"),
    days: int = Query(7, ge=1, le=365, description="Look back period in days"),
    db: Session = Depends(get_db)
) -> List[ThreatDetectionResponse]:
    """
    Retrieve list of detected threats with optional filtering.
    
    **Query Parameters:**
    - **limit**: Maximum number of results (1-100)
    - **offset**: Offset for pagination
    - **threat_level**: Filter by severity (CRITICAL, HIGH, MEDIUM, LOW, INFO)
    - **threat_type**: Filter by threat type (MALWARE, PHISHING, etc.)
    - **days**: Look back period in days (1-365)
    """
    try:
        logger.info(f"Fetching threats - limit: {limit}, offset: {offset}")
        
        # Build query
        query = db.query(Threat)
        
        # Filter by date
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        query = query.filter(Threat.detected_at >= cutoff_date)
        
        # Filter by threat level
        if threat_level:
            try:
                query = query.filter(Threat.threat_level == ThreatLevelEnum[threat_level])
            except KeyError:
                raise HTTPException(status_code=400, detail=f"Invalid threat level: {threat_level}")
        
        # Filter by threat type
        if threat_type:
            try:
                query = query.filter(Threat.threat_type == ThreatTypeEnum[threat_type])
            except KeyError:
                raise HTTPException(status_code=400, detail=f"Invalid threat type: {threat_type}")
        
        # Apply pagination
        threats = query.order_by(Threat.detected_at.desc()).offset(offset).limit(limit).all()
        
        # Convert to response models
        responses = []
        for threat in threats:
            responses.append(ThreatDetectionResponse(
                threat_id=threat.threat_id,
                threat_type=ThreatType[threat.threat_type.value],
                threat_level=ThreatLevel[threat.threat_level.value],
                detected_at=threat.detected_at,
                description=threat.description,
                indicators=[
                    {
                        "name": ind.name,
                        "value": ind.value,
                        "indicator_type": ind.indicator_type,
                        "confidence": ind.confidence
                    } for ind in threat.indicators
                ],
                affected_systems=[],
                recommendations=[],
                confidence=threat.confidence
            ))
        
        logger.info(f"Retrieved {len(threats)} threats")
        return responses
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving threats: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve threats")


@router.get(
    "/{threat_id}",
    response_model=ThreatDetectionResponse,
    summary="Get threat details by ID"
)
async def get_threat_details(
    threat_id: str = Query(..., min_length=1, description="Threat ID"),
    db: Session = Depends(get_db)
) -> ThreatDetectionResponse:
    """
    Retrieve detailed information about a specific threat.
    """
    try:
        logger.info(f"Fetching threat details: {threat_id}")
        
        threat = db.query(Threat).filter(Threat.threat_id == threat_id).first()
        if not threat:
            raise HTTPException(status_code=404, detail=f"Threat not found: {threat_id}")
        
        return ThreatDetectionResponse(
            threat_id=threat.threat_id,
            threat_type=ThreatType[threat.threat_type.value],
            threat_level=ThreatLevel[threat.threat_level.value],
            detected_at=threat.detected_at,
            description=threat.description,
            indicators=[
                {
                    "name": ind.name,
                    "value": ind.value,
                    "indicator_type": ind.indicator_type,
                    "confidence": ind.confidence
                } for ind in threat.indicators
            ],
            affected_systems=[],
            recommendations=[],
            confidence=threat.confidence
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving threat details: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve threat details")


# ==================== Helper Functions ====================

def _analyze_sample(sample: str, sample_type: str) -> Optional[dict]:
    """
    Mock threat analysis logic (replace with actual ML model).
    
    Args:
        sample: The sample to analyze
        sample_type: Type of sample (hash, url, ip, etc.)
    
    Returns:
        Dictionary with threat analysis or None if no threat detected
    """
    from schemas import ThreatIndicatorSchema, IndicatorType
    
    # Simulate threat detection based on sample patterns
    if any(keyword in sample.lower() for keyword in ["malware", "trojan", "virus"]):
        return {
            "type": ThreatType.MALWARE,
            "level": ThreatLevel.CRITICAL,
            "description": "Detected malicious code pattern",
            "confidence": 0.98,
            "indicators": [
                ThreatIndicatorSchema(
                    name="IOC_HASH",
                    value=sample,
                    indicator_type=IndicatorType.HASH,
                    confidence=0.98
                )
            ],
            "systems": ["SERVER-01", "WORKSTATION-05"],
            "recommendations": [
                "Isolate affected systems",
                "Scan with updated antivirus",
                "Review system logs"
            ],
            "details": {
                "file_type": "executable",
                "detected_families": ["Trojan.Generic"],
                "behavior_score": 95
            }
        }
    elif sample_type == "url" and any(x in sample for x in [".tk", ".ml", "phishing"]):
        return {
            "type": ThreatType.PHISHING,
            "level": ThreatLevel.HIGH,
            "description": "Suspected phishing URL",
            "confidence": 0.89,
            "indicators": [
                ThreatIndicatorSchema(
                    name="PHISHING_URL",
                    value=sample,
                    indicator_type=IndicatorType.URL,
                    confidence=0.89
                )
            ],
            "systems": [],
            "recommendations": [
                "Block URL in email gateway",
                "Warn users",
                "Report to phishing registry"
            ],
            "details": {
                "url_reputation": "malicious",
                "certificates": "invalid"
            }
        }
    
    return None
