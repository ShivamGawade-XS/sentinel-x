"""
SQLAlchemy models for Sentinel-X threat detection system.
Defines database schema for threats, events, reports, and analysis results.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, Integer, String, Text, Float, Boolean, DateTime,
    Enum as SQLEnum, ForeignKey, JSON, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum

Base = declarative_base()


class ThreatLevelEnum(str, enum.Enum):
    """Threat severity levels"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class ThreatTypeEnum(str, enum.Enum):
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


class AnalysisStatusEnum(str, enum.Enum):
    """Status of threat analysis"""
    PENDING = "PENDING"
    ANALYZING = "ANALYZING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Threat(Base):
    """
    Represents a detected threat in the system.
    """
    __tablename__ = "threats"
    __table_args__ = (
        UniqueConstraint('threat_id', name='uq_threat_id'),
    )

    id = Column(Integer, primary_key=True, index=True)
    threat_id = Column(String(64), unique=True, nullable=False, index=True)
    threat_type = Column(SQLEnum(ThreatTypeEnum), nullable=False, index=True)
    threat_level = Column(SQLEnum(ThreatLevelEnum), nullable=False, index=True)
    description = Column(Text, nullable=False)
    confidence = Column(Float, default=0.0)
    detected_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    indicators = relationship("ThreatIndicator", back_populates="threat", cascade="all, delete-orphan")
    affected_systems = relationship("AffectedSystem", back_populates="threat", cascade="all, delete-orphan")
    recommendations = relationship("ThreatRecommendation", back_populates="threat", cascade="all, delete-orphan")
    analysis_results = relationship("ThreatAnalysis", back_populates="threat")
    
    def __repr__(self):
        return f"<Threat(threat_id={self.threat_id}, type={self.threat_type}, level={self.threat_level})>"


class ThreatIndicator(Base):
    """
    Represents an indicator of compromise (IOC) associated with a threat.
    """
    __tablename__ = "threat_indicators"
    __table_args__ = (
        UniqueConstraint('threat_id', 'value', 'indicator_type', name='uq_threat_indicator'),
    )

    id = Column(Integer, primary_key=True, index=True)
    threat_id = Column(Integer, ForeignKey("threats.id"), nullable=False, index=True)
    name = Column(String(128), nullable=False)
    value = Column(String(512), nullable=False, index=True)
    indicator_type = Column(String(64), nullable=False, index=True)  # ip, hash, domain, url, etc.
    confidence = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    threat = relationship("Threat", back_populates="indicators")
    
    def __repr__(self):
        return f"<ThreatIndicator(name={self.name}, type={self.indicator_type})>"


class AffectedSystem(Base):
    """
    Represents a system affected by a threat.
    """
    __tablename__ = "affected_systems"
    __table_args__ = (
        UniqueConstraint('threat_id', 'system_name', name='uq_threat_system'),
    )

    id = Column(Integer, primary_key=True, index=True)
    threat_id = Column(Integer, ForeignKey("threats.id"), nullable=False, index=True)
    system_name = Column(String(256), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    threat = relationship("Threat", back_populates="affected_systems")
    
    def __repr__(self):
        return f"<AffectedSystem(system={self.system_name})>"


class ThreatRecommendation(Base):
    """
    Represents a recommended action for a threat.
    """
    __tablename__ = "threat_recommendations"
    __table_args__ = (
        UniqueConstraint('threat_id', 'recommendation', name='uq_threat_recommendation'),
    )

    id = Column(Integer, primary_key=True, index=True)
    threat_id = Column(Integer, ForeignKey("threats.id"), nullable=False, index=True)
    recommendation = Column(Text, nullable=False)
    priority = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    threat = relationship("Threat", back_populates="recommendations")
    
    def __repr__(self):
        return f"<ThreatRecommendation(threat_id={self.threat_id})>"


class ThreatAnalysis(Base):
    """
    Represents a detailed analysis of a threat or artifact.
    """
    __tablename__ = "threat_analysis"
    __table_args__ = (
        UniqueConstraint('analysis_id', name='uq_analysis_id'),
    )

    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(String(64), unique=True, nullable=False, index=True)
    threat_id = Column(Integer, ForeignKey("threats.id"), nullable=True)
    sample = Column(String(512), nullable=False, index=True)
    sample_type = Column(String(64), nullable=False)  # hash, url, ip, file, etc.
    status = Column(SQLEnum(AnalysisStatusEnum), default=AnalysisStatusEnum.PENDING, nullable=False, index=True)
    threat_detected = Column(Boolean, default=False)
    analysis_details = Column(JSON, default={})
    processing_time_ms = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    threat = relationship("Threat", back_populates="analysis_results")
    
    def __repr__(self):
        return f"<ThreatAnalysis(analysis_id={self.analysis_id}, status={self.status})>"


class ThreatReport(Base):
    """
    Represents a generated threat report.
    """
    __tablename__ = "threat_reports"
    __table_args__ = (
        UniqueConstraint('report_id', name='uq_report_id'),
    )

    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(String(64), unique=True, nullable=False, index=True)
    title = Column(String(256), nullable=False)
    total_threats = Column(Integer, default=0)
    critical_count = Column(Integer, default=0)
    high_count = Column(Integer, default=0)
    medium_count = Column(Integer, default=0)
    low_count = Column(Integer, default=0)
    summary = Column(Text)
    recommendations = Column(JSON, default=[])
    metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    def __repr__(self):
        return f"<ThreatReport(report_id={self.report_id}, title={self.title})>"


class SecurityEvent(Base):
    """
    Represents a security event detected in the system.
    """
    __tablename__ = "security_events"
    __table_args__ = (
        UniqueConstraint('event_id', name='uq_event_id'),
    )

    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(String(64), unique=True, nullable=False, index=True)
    event_type = Column(String(128), nullable=False, index=True)
    severity = Column(SQLEnum(ThreatLevelEnum), nullable=False, index=True)
    source = Column(String(256), nullable=False, index=True)
    description = Column(Text, nullable=False)
    metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    def __repr__(self):
        return f"<SecurityEvent(event_id={self.event_id}, type={self.event_type})>"


class AuditLog(Base):
    """
    Represents an audit log entry for tracking API access and changes.
    """
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    action = Column(String(128), nullable=False)
    resource_type = Column(String(64), nullable=False)
    resource_id = Column(String(256))
    user_id = Column(String(256))
    ip_address = Column(String(45))
    request_method = Column(String(10))
    request_path = Column(String(512))
    status_code = Column(Integer)
    details = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    def __repr__(self):
        return f"<AuditLog(action={self.action}, resource_type={self.resource_type})>"
