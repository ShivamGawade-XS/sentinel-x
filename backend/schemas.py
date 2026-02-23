"""
Pydantic schemas for request/response validation in Sentinel-X API.
Provides models for all API endpoints with comprehensive validation rules.
"""

from pydantic import BaseModel, Field, EmailStr, HttpUrl, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# ==================== Enums ====================

class ThreatLevel(str, Enum):
    """Threat severity levels"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class ThreatType(str, Enum):
    """Types of threats"""
    MALWARE = "MALWARE"
    PHISHING = "PHISHING"
    DDoS = "DDoS"
    SQL_INJECTION = "SQL_INJECTION"
    XSS = "XSS"
    RANSOMWARE = "RANSOMWARE"
    ZERO_DAY = "ZERO_DAY"
    SUSPICIOUS_ACTIVITY = "SUSPICIOUS_ACTIVITY"
    ANOMALY = "ANOMALY"
    UNKNOWN = "UNKNOWN"


class AnalysisStatus(str, Enum):
    """Status of threat analysis"""
    PENDING = "PENDING"
    ANALYZING = "ANALYZING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class IndicatorType(str, Enum):
    """Types of threat indicators"""
    IP = "IP"
    DOMAIN = "DOMAIN"
    URL = "URL"
    HASH = "HASH"
    EMAIL = "EMAIL"
    FILE_PATH = "FILE_PATH"
    PROCESS = "PROCESS"
    REGISTRY_KEY = "REGISTRY_KEY"
    CERTIFICATE = "CERTIFICATE"


# ==================== Shared Schemas ====================

class ThreatIndicatorSchema(BaseModel):
    """Schema for threat indicators"""
    name: str = Field(..., min_length=1, max_length=128, description="Name of the indicator")
    value: str = Field(..., min_length=1, max_length=512, description="Indicator value")
    indicator_type: IndicatorType = Field(..., description="Type of indicator")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "IOC_HASH",
                "value": "5d41402abc4b2a76b9719d911017c592",
                "indicator_type": "HASH",
                "confidence": 0.98
            }
        }


class AffectedSystemSchema(BaseModel):
    """Schema for affected systems"""
    system_name: str = Field(..., min_length=1, max_length=256, description="Name of affected system")
    
    class Config:
        json_schema_extra = {
            "example": {
                "system_name": "SERVER-01"
            }
        }


class ThreatRecommendationSchema(BaseModel):
    """Schema for threat recommendations"""
    recommendation: str = Field(..., min_length=1, description="Recommended action")
    priority: int = Field(default=0, ge=0, le=10, description="Priority level (0-10)")


# ==================== Request Schemas ====================

class ThreatDetectionRequest(BaseModel):
    """Schema for threat detection request"""
    sample: str = Field(..., min_length=1, max_length=512, description="Sample to analyze")
    sample_type: str = Field(default="hash", description="Type of sample (hash, url, ip, file)")
    priority: Optional[ThreatLevel] = Field(default=ThreatLevel.MEDIUM, description="Priority level")
    tags: Optional[List[str]] = Field(default=[], max_items=20, description="Custom tags")
    
    @field_validator('sample_type')
    @classmethod
    def validate_sample_type(cls, v):
        allowed_types = ['hash', 'url', 'ip', 'file', 'domain', 'email']
        if v not in allowed_types:
            raise ValueError(f'sample_type must be one of {allowed_types}')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "sample": "5d41402abc4b2a76b9719d911017c592",
                "sample_type": "hash",
                "priority": "HIGH",
                "tags": ["malware", "trojan"]
            }
        }


class SecurityEventRequest(BaseModel):
    """Schema for logging security events"""
    event_type: str = Field(..., min_length=1, max_length=128, description="Type of event")
    severity: ThreatLevel = Field(..., description="Event severity")
    source: str = Field(..., min_length=1, max_length=256, description="Event source (IP, host, etc.)")
    description: str = Field(..., min_length=1, description="Event description")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "event_type": "SUSPICIOUS_LOGIN",
                "severity": "HIGH",
                "source": "192.168.1.100",
                "description": "Multiple failed login attempts detected",
                "metadata": {"attempts": 5, "user": "admin"}
            }
        }


class ThreatReportRequest(BaseModel):
    """Schema for generating threat reports"""
    title: str = Field(..., min_length=1, max_length=256, description="Report title")
    threat_ids: Optional[List[str]] = Field(default=None, max_items=100, description="Specific threat IDs to include")
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "Weekly Security Report",
                "threat_ids": ["THR-001", "THR-002"]
            }
        }


# ==================== Response Schemas ====================

class ThreatDetectionResponse(BaseModel):
    """Schema for threat detection response"""
    threat_id: str = Field(..., description="Unique threat identifier")
    threat_type: ThreatType = Field(..., description="Type of threat detected")
    threat_level: ThreatLevel = Field(..., description="Severity level")
    detected_at: datetime = Field(..., description="Detection timestamp")
    description: str = Field(..., description="Threat description")
    indicators: List[ThreatIndicatorSchema] = Field(default=[], description="Associated indicators")
    affected_systems: List[str] = Field(default=[], description="Affected systems/hosts")
    recommendations: List[str] = Field(default=[], description="Recommended actions")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Detection confidence")
    
    class Config:
        from_attributes = True


class ThreatAnalysisResponse(BaseModel):
    """Schema for threat analysis result"""
    analysis_id: str = Field(..., description="Unique analysis identifier")
    status: AnalysisStatus = Field(..., description="Analysis status")
    sample: str = Field(..., description="Analyzed sample")
    threat_detected: bool = Field(..., description="Whether a threat was detected")
    threat: Optional[ThreatDetectionResponse] = None
    indicators: List[ThreatIndicatorSchema] = Field(default=[], description="Extracted indicators")
    analysis_details: Dict[str, Any] = Field(default={}, description="Detailed analysis information")
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: float = Field(default=0.0, description="Processing time in milliseconds")
    
    class Config:
        from_attributes = True


class ThreatReportResponse(BaseModel):
    """Schema for threat report response"""
    report_id: str = Field(..., description="Unique report identifier")
    title: str = Field(..., description="Report title")
    threats: List[ThreatDetectionResponse] = Field(..., description="List of detected threats")
    total_threats: int = Field(..., ge=0, description="Total number of threats")
    critical_count: int = Field(default=0, ge=0, description="Count of critical threats")
    high_count: int = Field(default=0, ge=0, description="Count of high severity threats")
    medium_count: int = Field(default=0, ge=0, description="Count of medium severity threats")
    low_count: int = Field(default=0, ge=0, description="Count of low severity threats")
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    summary: str = Field(..., description="Executive summary")
    recommendations: List[str] = Field(default=[], description="Overall recommendations")
    
    class Config:
        from_attributes = True


class SecurityEventResponse(BaseModel):
    """Schema for security event response"""
    event_id: str = Field(..., description="Unique event identifier")
    event_type: str = Field(..., description="Type of security event")
    severity: ThreatLevel = Field(..., description="Event severity")
    source: str = Field(..., description="Event source")
    description: str = Field(..., description="Event description")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")
    
    class Config:
        from_attributes = True


class HealthCheckResponse(BaseModel):
    """Schema for health check response"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Check timestamp")
    database: bool = Field(..., description="Database connectivity status")
    version: str = Field(..., description="API version")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-01T12:00:00",
                "database": True,
                "version": "1.0.0"
            }
        }


class ErrorResponse(BaseModel):
    """Schema for error responses"""
    status: str = Field(default="error", description="Error status")
    code: int = Field(..., description="HTTP status code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "error",
                "code": 400,
                "message": "Invalid request parameters",
                "timestamp": "2024-01-01T12:00:00"
            }
        }


class PaginationParams(BaseModel):
    """Schema for pagination parameters"""
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of results")
    offset: int = Field(default=0, ge=0, description="Offset for pagination")
    
    class Config:
        json_schema_extra = {
            "example": {
                "limit": 20,
                "offset": 0
            }
        }


class StatisticsResponse(BaseModel):
    """Schema for statistics response"""
    total_threats: int = Field(..., description="Total number of threats")
    threats_today: int = Field(..., description="Threats detected today")
    threats_this_week: int = Field(..., description="Threats detected this week")
    critical_threats: int = Field(..., description="Count of critical threats")
    high_severity: int = Field(..., description="Count of high severity threats")
    medium_severity: int = Field(..., description="Count of medium severity threats")
    low_severity: int = Field(..., description="Count of low severity threats")
    top_threat_types: Dict[str, int] = Field(..., description="Top threat types by count")
    avg_detection_time_ms: float = Field(..., description="Average detection time in milliseconds")
    detection_rate_percentage: float = Field(..., ge=0, le=100, description="Detection rate percentage")
    
    class Config:
        from_attributes = True
