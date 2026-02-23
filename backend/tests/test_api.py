"""
Test suite for Sentinel-X API endpoints.
Tests threat detection, event logging, and statistics endpoints.
"""

import pytest
from tests.conftest import client


class TestHealthCheck:
    """Tests for health check endpoints"""
    
    def test_health_check_root(self, client):
        """Test root health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "timestamp" in data
        assert "version" in data
    
    def test_health_check_v1(self, client):
        """Test API v1 health check endpoint"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "database" in data


class TestThreatDetection:
    """Tests for threat detection endpoints"""
    
    def test_detect_threat_malware_hash(self, client):
        """Test detecting malware by hash"""
        payload = {
            "sample": "malware_hash_5d41402abc",
            "sample_type": "hash",
            "priority": "HIGH",
            "tags": ["suspicious", "malware"]
        }
        response = client.post("/api/v1/threats/detect", json=payload)
        # Should fail since it's a test hash not matching malware patterns
        assert response.status_code in [200, 404]
    
    def test_detect_threat_with_malware_keyword(self, client):
        """Test detecting sample with malware keyword"""
        payload = {
            "sample": "malware_detected_pattern",
            "sample_type": "hash",
            "priority": "CRITICAL"
        }
        response = client.post("/api/v1/threats/detect", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["threat_type"] == "MALWARE"
        assert data["threat_level"] == "CRITICAL"
    
    def test_detect_threat_phishing_url(self, client):
        """Test detecting phishing URL"""
        payload = {
            "sample": "http://phishing.tk/malicious",
            "sample_type": "url"
        }
        response = client.post("/api/v1/threats/detect", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["threat_type"] == "PHISHING"
        assert data["threat_level"] == "HIGH"
    
    def test_detect_threat_invalid_sample_type(self, client):
        """Test detection with invalid sample type"""
        payload = {
            "sample": "test",
            "sample_type": "invalid_type"
        }
        response = client.post("/api/v1/threats/detect", json=payload)
        assert response.status_code == 422  # Validation error


class TestThreatAnalysis:
    """Tests for threat analysis endpoints"""
    
    def test_analyze_threat_with_malware(self, client):
        """Test analyzing sample with malware"""
        payload = {
            "sample": "trojan_pattern_detected",
            "sample_type": "hash"
        }
        response = client.post("/api/v1/threats/analyze", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "analysis_id" in data
        assert data["status"] == "COMPLETED"
        assert data["threat_detected"] is True
    
    def test_analyze_threat_clean_sample(self, client):
        """Test analyzing clean sample"""
        payload = {
            "sample": "clean_file_hash_abc123",
            "sample_type": "hash"
        }
        response = client.post("/api/v1/threats/analyze", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["threat_detected"] is False


class TestSecurityEvents:
    """Tests for security event endpoints"""
    
    def test_log_security_event(self, client):
        """Test logging a security event"""
        payload = {
            "event_type": "SUSPICIOUS_LOGIN",
            "severity": "HIGH",
            "source": "192.168.1.100",
            "description": "Multiple failed login attempts detected",
            "metadata": {"attempts": 5, "user": "admin"}
        }
        response = client.post("/api/v1/events/log", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert "event_id" in data
        assert data["source"] == "192.168.1.100"
        assert data["event_type"] == "SUSPICIOUS_LOGIN"
    
    def test_get_security_events(self, client):
        """Test retrieving security events"""
        # First log an event
        payload = {
            "event_type": "NETWORK_ANOMALY",
            "severity": "MEDIUM",
            "source": "192.168.1.50",
            "description": "Unusual network traffic"
        }
        client.post("/api/v1/events/log", json=payload)
        
        # Then retrieve events
        response = client.get("/api/v1/events?limit=10&offset=0")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_security_events_with_filter(self, client):
        """Test retrieving events with type filter"""
        response = client.get("/api/v1/events?event_type=LOGIN")
        assert response.status_code == 200


class TestStatistics:
    """Tests for statistics endpoints"""
    
    def test_get_threat_statistics(self, client):
        """Test retrieving threat statistics"""
        response = client.get("/api/v1/statistics/threats?days=30")
        assert response.status_code == 200
        data = response.json()
        assert "total_threats" in data
        assert "threats_today" in data
        assert "threats_this_week" in data
        assert "critical_threats" in data
        assert "top_threat_types" in data
    
    def test_get_threats_by_type(self, client):
        """Test getting threat distribution by type"""
        response = client.get("/api/v1/statistics/threats/by-type?days=30")
        assert response.status_code == 200
        data = response.json()
        assert "by_type" in data
        assert "period_days" in data
    
    def test_get_threats_by_severity(self, client):
        """Test getting threat distribution by severity"""
        response = client.get("/api/v1/statistics/threats/by-severity?days=30")
        assert response.status_code == 200
        data = response.json()
        assert "by_severity" in data
    
    def test_get_system_overview(self, client):
        """Test getting system overview"""
        response = client.get("/api/v1/statistics/system/overview")
        assert response.status_code == 200
        data = response.json()
        assert "all_time" in data
        assert "last_24_hours" in data
        assert "active_alerts" in data
    
    def test_get_threat_detection_trends(self, client):
        """Test getting threat detection trends"""
        response = client.get("/api/v1/statistics/trends/threat-detection?days=7")
        assert response.status_code == 200
        data = response.json()
        assert "daily_threats" in data
        assert "total_threats" in data


class TestEventStatistics:
    """Tests for event statistics endpoints"""
    
    def test_get_events_by_type(self, client):
        """Test getting event distribution by type"""
        response = client.get("/api/v1/events/stats/by-type?days=7")
        assert response.status_code == 200
        data = response.json()
        assert "by_type" in data
        assert "total_events" in data
    
    def test_get_events_by_severity(self, client):
        """Test getting event distribution by severity"""
        response = client.get("/api/v1/events/stats/by-severity?days=7")
        assert response.status_code == 200
        data = response.json()
        assert "by_severity" in data


class TestErrorHandling:
    """Tests for error handling"""
    
    def test_missing_required_field(self, client):
        """Test validation error for missing required field"""
        payload = {
            "sample_type": "hash"
            # Missing 'sample' field
        }
        response = client.post("/api/v1/threats/detect", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_invalid_json(self, client):
        """Test error handling for invalid JSON"""
        response = client.post(
            "/api/v1/threats/detect",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code in [400, 422]
    
    def test_nonexistent_threat(self, client):
        """Test retrieving nonexistent threat"""
        response = client.get("/api/v1/threats/THR-NONEXISTENT")
        assert response.status_code == 404
