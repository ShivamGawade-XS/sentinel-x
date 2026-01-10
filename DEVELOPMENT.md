# Sentinel-X Development Guide

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Setup Instructions](#setup-instructions)
4. [Module Descriptions](#module-descriptions)
5. [API Documentation](#api-documentation)
6. [ML Models](#ml-models)
7. [Testing Procedures](#testing-procedures)
8. [Deployment Instructions](#deployment-instructions)
9. [Contributing Guidelines](#contributing-guidelines)
10. [Troubleshooting](#troubleshooting)

---

## Overview

**Sentinel-X** is an intelligent threat detection and security monitoring system designed to provide real-time threat intelligence, anomaly detection, and security analytics. This development guide provides comprehensive documentation for developers working on the Sentinel-X platform.

### Project Goals
- Real-time threat detection and analysis
- Advanced anomaly detection using machine learning
- Comprehensive security event logging and analysis
- Scalable and extensible architecture
- Easy integration with existing security infrastructure

### Technology Stack
- **Backend**: Python 3.9+
- **Framework**: FastAPI
- **Database**: PostgreSQL
- **Message Queue**: Redis/RabbitMQ
- **ML Framework**: TensorFlow/PyTorch
- **Frontend**: React.js
- **Containerization**: Docker
- **Orchestration**: Kubernetes

---

## Architecture

### System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Applications                      │
└────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                    API Gateway (FastAPI)                     │
│         (/api/v1/auth, /api/v1/threats, /api/v1/alerts)    │
└────────────────────────────────────────────────────────────┘
                             ↓
        ┌────────────────────┼────────────────────┐
        ↓                    ↓                    ↓
   ┌─────────────┐   ┌──────────────┐   ┌────────────────┐
   │ Auth Module │   │ Threat Module│   │ Analytics Mod. │
   └─────────────┘   └──────────────┘   └────────────────┘
        ↓                    ↓                    ↓
   ┌─────────────┐   ┌──────────────┐   ┌────────────────┐
   │  JWT Store  │   │ ML Pipeline  │   │ Time Series DB │
   └─────────────┘   └──────────────┘   └────────────────┘
                             ↓
        ┌────────────────────┼────────────────────┐
        ↓                    ↓                    ↓
   ┌─────────────┐   ┌──────────────┐   ┌────────────────┐
   │ PostgreSQL  │   │   Redis      │   │  Elasticsearch │
   └─────────────┘   └──────────────┘   └────────────────┘
```

### Core Components

1. **API Gateway**: Central entry point for all client requests
2. **Authentication Module**: Handles user authentication and authorization
3. **Threat Detection Module**: Core threat analysis and detection engine
4. **ML Pipeline**: Machine learning models for anomaly detection
5. **Analytics Engine**: Data aggregation and reporting
6. **Event Processor**: Real-time event processing and correlation
7. **Database Layer**: Persistent data storage and caching

### Design Patterns Used
- **Microservices Architecture**: Modular, independently deployable services
- **Event-Driven Architecture**: Asynchronous processing using message queues
- **Repository Pattern**: Abstraction layer for data access
- **Dependency Injection**: Loose coupling through dependency injection
- **Factory Pattern**: Dynamic object creation for models and handlers

---

## Setup Instructions

### Prerequisites

- **Python 3.9+**: Ensure Python is installed
- **PostgreSQL 13+**: Database server
- **Redis 6+**: Caching and message broker
- **Docker & Docker Compose**: Containerization
- **Git**: Version control
- **Node.js 16+**: Frontend development (if applicable)

### Local Development Setup

#### 1. Clone the Repository

```bash
git clone https://github.com/ShivamGawade-XS/sentinel-x.git
cd sentinel-x
```

#### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development tools
```

#### 4. Environment Configuration

Create a `.env` file in the project root:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/sentinel_x_dev
SQLALCHEMY_ECHO=True

# Redis
REDIS_URL=redis://localhost:6379/0

# JWT
SECRET_KEY=your-secret-key-here-min-32-chars
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Logging
LOG_LEVEL=DEBUG
LOG_FILE=logs/sentinel-x.log

# ML Models
MODEL_PATH=./models/
USE_GPU=False

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_TITLE=Sentinel-X API
API_VERSION=1.0.0

# Security
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8000"]
```

#### 5. Database Setup

```bash
# Create database and tables
alembic upgrade head

# Seed initial data (optional)
python scripts/seed_database.py
```

#### 6. Run Development Server

```bash
# Start FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# In another terminal, start Celery worker
celery -A app.tasks worker --loglevel=info

# Start Celery beat scheduler
celery -A app.tasks beat --loglevel=info
```

### Using Docker Compose

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Verify Installation

```bash
# Check API is running
curl http://localhost:8000/api/v1/health

# Run test suite
pytest tests/ -v
```

---

## Module Descriptions

### 1. Authentication Module (`app/auth/`)

**Purpose**: Handle user authentication, authorization, and token management

**Key Files**:
- `models.py`: User models and permission definitions
- `schemas.py`: Request/response schemas
- `utils.py`: JWT token generation and verification
- `routes.py`: Authentication endpoints
- `dependencies.py`: Dependency injection for auth

**Key Functions**:
```python
- create_access_token(data: dict) -> str
- verify_token(token: str) -> dict
- get_current_user(token: str) -> User
- check_permissions(user: User, required: str) -> bool
```

**API Endpoints**:
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/refresh` - Refresh token
- `POST /api/v1/auth/logout` - User logout

### 2. Threat Detection Module (`app/threats/`)

**Purpose**: Core threat detection and analysis engine

**Key Files**:
- `models.py`: Threat data models
- `detector.py`: Threat detection algorithms
- `classifier.py`: Threat classification logic
- `enricher.py`: Threat intelligence enrichment
- `routes.py`: API endpoints

**Key Components**:
```python
class ThreatDetector:
    - detect_anomalies(data) -> List[Threat]
    - classify_threat(threat) -> str
    - correlate_events(events) -> List[CorrelatedThreat]
    - calculate_risk_score(threat) -> float
```

**API Endpoints**:
- `GET /api/v1/threats` - List all threats
- `POST /api/v1/threats/analyze` - Analyze raw data for threats
- `GET /api/v1/threats/{id}` - Get threat details
- `PUT /api/v1/threats/{id}/status` - Update threat status

### 3. Event Processing Module (`app/events/`)

**Purpose**: Real-time event ingestion and processing

**Key Files**:
- `schemas.py`: Event data structures
- `processor.py`: Event processing logic
- `correlator.py`: Event correlation engine
- `routes.py`: Event ingestion endpoints

**Key Features**:
- High-throughput event ingestion
- Real-time correlation
- Event enrichment and normalization
- Time-series event analysis

**API Endpoints**:
- `POST /api/v1/events/ingest` - Ingest events
- `POST /api/v1/events/batch` - Batch event ingestion
- `GET /api/v1/events` - Query events

### 4. Analytics Module (`app/analytics/`)

**Purpose**: Data aggregation, reporting, and insights generation

**Key Files**:
- `aggregator.py`: Time-series data aggregation
- `reporter.py`: Report generation
- `dashboard.py`: Dashboard data providers
- `routes.py`: Analytics API endpoints

**Reports Generated**:
- Daily threat summary
- Risk trend analysis
- Threat category breakdown
- Top threat sources/targets
- Detection rate metrics

**API Endpoints**:
- `GET /api/v1/analytics/dashboard` - Dashboard data
- `GET /api/v1/analytics/reports/{type}` - Generate reports
- `GET /api/v1/analytics/trends` - Trend analysis

### 5. Database Module (`app/database/`)

**Purpose**: Database connection, session management, and ORM models

**Key Files**:
- `session.py`: SQLAlchemy session configuration
- `base.py`: Base model for ORM
- `models.py`: Core data models
- `migrations/`: Alembic migration scripts

**Models**:
- User, Role, Permission
- Threat, ThreatIndicator
- Event, EventLog
- Alert, AlertRule
- MLModel, ModelVersion

### 6. ML Pipeline Module (`app/ml/`)

**Purpose**: Machine learning model management and inference

**Key Files**:
- `models/`: Pre-trained model storage
- `trainer.py`: Model training logic
- `predictor.py`: Inference engine
- `evaluator.py`: Model evaluation metrics
- `feature_engineering.py`: Feature extraction

**Available Models**:
- Anomaly Detection Model
- Threat Classification Model
- Risk Scoring Model
- Pattern Recognition Model

### 7. Utilities Module (`app/utils/`)

**Purpose**: Shared utility functions

**Key Modules**:
- `logger.py`: Logging configuration
- `decorators.py`: Custom decorators
- `validators.py`: Data validation
- `helpers.py`: Helper functions
- `constants.py`: Application constants

---

## API Documentation

### Base URL
```
http://localhost:8000/api/v1
```

### Authentication

All endpoints require JWT token in Authorization header:
```
Authorization: Bearer <token>
```

### Common Response Format

**Success Response**:
```json
{
  "status": "success",
  "data": {...},
  "message": "Operation completed successfully"
}
```

**Error Response**:
```json
{
  "status": "error",
  "error_code": "INVALID_REQUEST",
  "message": "Description of error",
  "details": {...}
}
```

### Authentication Endpoints

#### Register User
```
POST /auth/register
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "securepassword",
  "full_name": "John Doe",
  "organization": "Acme Corp"
}

Response (201):
{
  "status": "success",
  "data": {
    "user_id": "uuid",
    "username": "user@example.com",
    "full_name": "John Doe",
    "created_at": "2026-01-10T05:44:29Z"
  }
}
```

#### Login
```
POST /auth/login
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "securepassword"
}

Response (200):
{
  "status": "success",
  "data": {
    "access_token": "eyJhbGc...",
    "token_type": "bearer",
    "expires_in": 1800,
    "user": {
      "user_id": "uuid",
      "username": "user@example.com"
    }
  }
}
```

### Threat Endpoints

#### Get All Threats
```
GET /threats?skip=0&limit=10&severity=HIGH&status=OPEN

Response (200):
{
  "status": "success",
  "data": [
    {
      "threat_id": "uuid",
      "type": "malware",
      "severity": "HIGH",
      "status": "OPEN",
      "detected_at": "2026-01-10T05:44:29Z",
      "source_ip": "192.168.1.1",
      "target_ip": "10.0.0.1",
      "risk_score": 8.5,
      "indicators": ["indicator1", "indicator2"]
    }
  ],
  "total": 42,
  "skip": 0,
  "limit": 10
}
```

#### Analyze Data for Threats
```
POST /threats/analyze
Content-Type: application/json

{
  "source_ip": "192.168.1.100",
  "target_ip": "8.8.8.8",
  "port": 443,
  "protocol": "HTTPS",
  "bytes_sent": 10240,
  "bytes_received": 20480,
  "connection_duration_seconds": 120,
  "packet_count": 256
}

Response (200):
{
  "status": "success",
  "data": {
    "is_threat": true,
    "threat_type": "suspicious_outbound",
    "risk_score": 6.7,
    "confidence": 0.85,
    "indicators": [
      {
        "name": "high_volume_transfer",
        "value": 30720,
        "threshold": 20000
      }
    ],
    "recommendations": [
      "Monitor additional traffic from this source",
      "Review target IP reputation"
    ]
  }
}
```

#### Get Threat Details
```
GET /threats/{threat_id}

Response (200):
{
  "status": "success",
  "data": {
    "threat_id": "uuid",
    "type": "malware",
    "severity": "HIGH",
    "status": "OPEN",
    "detected_at": "2026-01-10T05:44:29Z",
    "source_ip": "192.168.1.1",
    "target_ip": "10.0.0.1",
    "risk_score": 8.5,
    "description": "Detected suspicious process behavior",
    "indicators": [...],
    "events": [...],
    "recommendations": [...],
    "updated_at": "2026-01-10T05:44:29Z"
  }
}
```

#### Update Threat Status
```
PUT /threats/{threat_id}/status
Content-Type: application/json

{
  "status": "RESOLVED",
  "resolution_notes": "False positive - whitelisted process"
}

Response (200):
{
  "status": "success",
  "message": "Threat status updated successfully"
}
```

### Event Endpoints

#### Ingest Single Event
```
POST /events/ingest
Content-Type: application/json

{
  "event_type": "network_traffic",
  "timestamp": "2026-01-10T05:44:29Z",
  "source_ip": "192.168.1.1",
  "destination_ip": "8.8.8.8",
  "port": 443,
  "protocol": "HTTPS",
  "bytes_transferred": 10240,
  "metadata": {
    "hostname": "workstation-01",
    "user": "john.doe",
    "process": "chrome.exe"
  }
}

Response (202):
{
  "status": "success",
  "data": {
    "event_id": "uuid",
    "status": "processed"
  }
}
```

#### Batch Ingest Events
```
POST /events/batch
Content-Type: application/json

{
  "events": [
    {...event1...},
    {...event2...},
    {...event3...}
  ]
}

Response (202):
{
  "status": "success",
  "data": {
    "total_received": 3,
    "total_processed": 3,
    "failed": 0
  }
}
```

### Analytics Endpoints

#### Get Dashboard Data
```
GET /analytics/dashboard?time_range=24h

Response (200):
{
  "status": "success",
  "data": {
    "summary": {
      "total_threats": 42,
      "high_severity": 5,
      "open_alerts": 12,
      "threat_trend": 15.5
    },
    "threat_by_type": {
      "malware": 15,
      "intrusion": 8,
      "anomaly": 12,
      "policy_violation": 7
    },
    "threat_by_severity": {
      "CRITICAL": 2,
      "HIGH": 5,
      "MEDIUM": 20,
      "LOW": 15
    },
    "top_sources": [
      {"ip": "192.168.1.5", "threats": 8},
      {"ip": "10.0.0.20", "threats": 6}
    ],
    "recent_threats": [...]
  }
}
```

### Status Codes

- `200`: Success
- `201`: Created
- `202`: Accepted (async processing)
- `400`: Bad Request
- `401`: Unauthorized
- `403`: Forbidden
- `404`: Not Found
- `409`: Conflict
- `500`: Internal Server Error
- `503`: Service Unavailable

---

## ML Models

### Overview

Sentinel-X uses multiple machine learning models for threat detection and analysis. All models are versioned and can be trained, evaluated, and deployed independently.

### Available Models

#### 1. Anomaly Detection Model

**Purpose**: Identify unusual network behavior patterns

**Type**: Unsupervised Learning (Isolation Forest, Autoencoder)

**Input Features**:
- Source IP address
- Destination IP address
- Port number
- Protocol type
- Bytes sent/received
- Connection duration
- Packet count
- Time of day
- Day of week

**Output**: Anomaly score (0-1)

**Training**:
```bash
python scripts/train_anomaly_model.py \
  --data_path data/normal_traffic.csv \
  --test_size 0.2 \
  --model_type isolation_forest \
  --save_model models/anomaly_detection_v1.pkl
```

**Inference**:
```python
from app.ml.predictor import AnomalyPredictor

predictor = AnomalyPredictor.load("models/anomaly_detection_v1.pkl")
features = {
    "source_ip": "192.168.1.1",
    "destination_ip": "8.8.8.8",
    "bytes_sent": 10240,
    ...
}
anomaly_score = predictor.predict(features)
```

#### 2. Threat Classification Model

**Purpose**: Classify detected threats into specific categories

**Type**: Supervised Learning (Random Forest, Gradient Boosting)

**Classes**:
- `malware`: Malicious software detected
- `intrusion`: Unauthorized access attempt
- `anomaly`: Unusual behavior
- `policy_violation`: Policy compliance issue
- `reconnaissance`: Information gathering activity
- `exfiltration`: Data exfiltration attempt

**Training**:
```bash
python scripts/train_classifier_model.py \
  --data_path data/labeled_threats.csv \
  --test_size 0.2 \
  --model_type random_forest \
  --save_model models/classifier_v1.pkl
```

**Evaluation Metrics**:
- Accuracy
- Precision per class
- Recall per class
- F1-score
- Confusion matrix
- ROC-AUC

#### 3. Risk Scoring Model

**Purpose**: Calculate overall risk score for threats

**Type**: Ensemble Model

**Factors**:
- Threat severity
- Target criticality
- Source reputation
- Attack vector
- Historical patterns
- Indicator count

**Output**: Risk score (0-10)

#### 4. Pattern Recognition Model

**Purpose**: Identify attack patterns and campaigns

**Type**: Unsupervised/Semi-supervised (Clustering, Hidden Markov Model)

**Use Cases**:
- Identify coordinated attacks
- Detect attack campaigns
- Find attack patterns
- Group related events

### Model Training Pipeline

```bash
# Full pipeline
python scripts/train_all_models.py \
  --data_path data/ \
  --validate \
  --save_results results/

# Specific model training with cross-validation
python scripts/train_model.py \
  --model_type anomaly_detection \
  --cv_folds 5 \
  --hyperparameter_tuning \
  --save_model models/latest/
```

### Model Evaluation

```bash
# Evaluate models
python scripts/evaluate_models.py \
  --models_path models/ \
  --test_data data/test_set.csv \
  --output_report results/evaluation_report.html

# Generate confusion matrix
python scripts/generate_reports.py \
  --model_path models/classifier_v1.pkl \
  --test_data data/test_set.csv \
  --report_type confusion_matrix
```

### Model Versioning

```python
# Access model registry
from app.ml.registry import ModelRegistry

registry = ModelRegistry()

# Get latest model
model = registry.get_latest_model("anomaly_detection")

# Get specific version
model_v1 = registry.get_model("anomaly_detection", version="1.0.0")

# List all versions
versions = registry.list_versions("anomaly_detection")

# Register new model
registry.register_model(
    model_name="anomaly_detection",
    model_object=trained_model,
    version="2.0.0",
    metrics={"accuracy": 0.94, "f1_score": 0.91}
)
```

### Model Performance Monitoring

```python
# Monitor model drift
from app.ml.monitoring import ModelMonitor

monitor = ModelMonitor()

# Check if retraining is needed
needs_retraining = monitor.check_model_drift(
    model="anomaly_detection",
    threshold=0.05
)

# Get performance metrics
metrics = monitor.get_performance_metrics(
    model="classifier",
    time_range="7d"
)
```

---

## Testing Procedures

### Test Structure

```
tests/
├── unit/
│   ├── test_auth.py
│   ├── test_threats.py
│   ├── test_events.py
│   ├── test_ml_models.py
│   └── test_utils.py
├── integration/
│   ├── test_auth_flow.py
│   ├── test_threat_detection_flow.py
│   ├── test_api_endpoints.py
│   └── test_database.py
├── performance/
│   ├── test_throughput.py
│   ├── test_latency.py
│   └── test_scalability.py
├── fixtures/
│   ├── conftest.py
│   ├── sample_data.py
│   └── mock_data.py
└── e2e/
    └── test_full_workflow.py
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/unit/test_auth.py -v

# Run specific test class
pytest tests/unit/test_auth.py::TestAuthentication -v

# Run specific test
pytest tests/unit/test_auth.py::TestAuthentication::test_login -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Run with markers
pytest tests/ -m "unit" -v
pytest tests/ -m "integration" -v
pytest tests/ -m "slow" -v

# Run in parallel
pytest tests/ -n auto
```

### Unit Tests Example

```python
# tests/unit/test_threats.py
import pytest
from app.threats.detector import ThreatDetector
from app.threats.models import Threat

@pytest.fixture
def threat_detector():
    return ThreatDetector()

class TestThreatDetection:
    
    def test_detect_malware(self, threat_detector):
        """Test malware detection"""
        suspicious_data = {
            "source_ip": "192.168.1.100",
            "destination_ip": "10.0.0.1",
            "bytes_sent": 50000,
            "process_name": "unknown.exe"
        }
        
        result = threat_detector.detect(suspicious_data)
        
        assert result.is_threat == True
        assert result.threat_type == "malware"
        assert result.risk_score > 0.7

    def test_classify_threat(self, threat_detector):
        """Test threat classification"""
        threat = Threat(
            type="network",
            indicators=["high_volume", "suspicious_port"]
        )
        
        classification = threat_detector.classify(threat)
        
        assert classification in ["malware", "intrusion", "anomaly"]
```

### Integration Tests Example

```python
# tests/integration/test_threat_detection_flow.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    return TestClient(app)

class TestThreatDetectionFlow:
    
    def test_analyze_and_detect_threat(self, client, authenticated_headers):
        """Test end-to-end threat detection flow"""
        
        # Ingest event
        event_response = client.post(
            "/api/v1/events/ingest",
            json={
                "source_ip": "192.168.1.1",
                "destination_ip": "8.8.8.8",
                "bytes_sent": 100000,
                "bytes_received": 50000
            },
            headers=authenticated_headers
        )
        assert event_response.status_code == 202
        
        # Analyze threat
        threat_response = client.post(
            "/api/v1/threats/analyze",
            json=event_response.json()["data"],
            headers=authenticated_headers
        )
        assert threat_response.status_code == 200
        
        threat_data = threat_response.json()["data"]
        assert threat_data["is_threat"] == True
        assert threat_data["risk_score"] > 0
```

### Performance Tests Example

```python
# tests/performance/test_throughput.py
import pytest
import time
from locust import HttpUser, task, between

class ThreatAnalysisUser(HttpUser):
    wait_time = between(1, 5)
    
    @task
    def analyze_threat(self):
        self.client.post(
            "/api/v1/threats/analyze",
            json={
                "source_ip": "192.168.1.1",
                "destination_ip": "8.8.8.8",
                "bytes_sent": 10240
            },
            headers={"Authorization": f"Bearer {self.token}"}
        )
```

### Test Configuration

```python
# tests/conftest.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database.session import get_db
from app.main import app

@pytest.fixture(scope="session")
def test_db():
    engine = create_engine("sqlite:///./test.db")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()

@pytest.fixture
def authenticated_headers(client, test_user):
    response = client.post(
        "/api/v1/auth/login",
        json={"username": test_user.username, "password": "testpass"}
    )
    token = response.json()["data"]["access_token"]
    return {"Authorization": f"Bearer {token}"}
```

### Testing Best Practices

1. **Isolation**: Each test should be independent
2. **Fixtures**: Use pytest fixtures for setup and teardown
3. **Mocking**: Mock external dependencies
4. **Clarity**: Use descriptive test names
5. **Coverage**: Aim for 80%+ code coverage
6. **Speed**: Keep unit tests fast
7. **Documentation**: Document complex test scenarios

---

## Deployment Instructions

### Pre-Deployment Checklist

- [ ] All tests passing
- [ ] Code review completed
- [ ] Security scan passed
- [ ] Performance tests successful
- [ ] Database migrations tested
- [ ] Dependencies updated
- [ ] Documentation updated
- [ ] Environment variables configured

### Environment Preparation

#### Development Environment
```bash
# Variables in .env.development
DEBUG=True
LOG_LEVEL=DEBUG
DB_POOL_SIZE=5
```

#### Staging Environment
```bash
# Variables in .env.staging
DEBUG=False
LOG_LEVEL=INFO
DB_POOL_SIZE=20
CORS_ORIGINS=["https://staging.example.com"]
```

#### Production Environment
```bash
# Variables in .env.production
DEBUG=False
LOG_LEVEL=WARNING
DB_POOL_SIZE=50
CORS_ORIGINS=["https://example.com"]
SECURE_SSL_REDIRECT=True
HTTPS_ONLY=True
```

### Docker Deployment

#### Build Docker Image

```bash
# Build development image
docker build -t sentinel-x:dev -f Dockerfile.dev .

# Build production image
docker build -t sentinel-x:latest -f Dockerfile .

# Build specific version
docker build -t sentinel-x:v1.0.0 .

# Build with build args
docker build \
  --build-arg PYTHON_VERSION=3.10 \
  --build-arg ENV=production \
  -t sentinel-x:latest .
```

#### Push to Registry

```bash
# Tag image
docker tag sentinel-x:latest gcr.io/project-id/sentinel-x:latest

# Push to registry
docker push gcr.io/project-id/sentinel-x:latest
```

#### Run Container

```bash
# Run single container
docker run -d \
  --name sentinel-x \
  -e DATABASE_URL=postgresql://user:pass@db:5432/sentinel_x \
  -e REDIS_URL=redis://redis:6379 \
  -p 8000:8000 \
  sentinel-x:latest

# Run with docker-compose
docker-compose -f docker-compose.yml up -d
```

### Kubernetes Deployment

#### Prepare Kubernetes Manifests

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentinel-x-api
  labels:
    app: sentinel-x
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sentinel-x
  template:
    metadata:
      labels:
        app: sentinel-x
    spec:
      containers:
      - name: sentinel-x
        image: gcr.io/project-id/sentinel-x:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: sentinel-x-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: sentinel-x-secrets
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/v1/ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: sentinel-x-service
spec:
  selector:
    app: sentinel-x
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

#### Deploy to Kubernetes

```bash
# Create namespace
kubectl create namespace sentinel-x

# Create secrets
kubectl create secret generic sentinel-x-secrets \
  --from-literal=database-url=$DATABASE_URL \
  --from-literal=redis-url=$REDIS_URL \
  -n sentinel-x

# Apply manifests
kubectl apply -f k8s/deployment.yaml -n sentinel-x

# Verify deployment
kubectl get deployment -n sentinel-x
kubectl get pods -n sentinel-x
kubectl get svc -n sentinel-x

# View logs
kubectl logs -f deployment/sentinel-x-api -n sentinel-x
```

### Database Migrations

```bash
# Generate migration
alembic revision --autogenerate -m "Add new column"

# Review migration
cat alembic/versions/xxx_add_new_column.py

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1

# Show migration history
alembic history
```

### Health Checks and Monitoring

```bash
# Health check endpoint
curl http://localhost:8000/api/v1/health

# Readiness check
curl http://localhost:8000/api/v1/ready

# Metrics endpoint
curl http://localhost:8000/metrics
```

### Rollback Procedure

```bash
# If deployment fails:

# 1. Check rollout status
kubectl rollout status deployment/sentinel-x-api -n sentinel-x

# 2. Rollback to previous version
kubectl rollout undo deployment/sentinel-x-api -n sentinel-x

# 3. Verify rollback
kubectl get deployment -n sentinel-x
kubectl logs -f deployment/sentinel-x-api -n sentinel-x

# 4. Investigate failure
kubectl get events -n sentinel-x
```

### CI/CD Pipeline

#### GitHub Actions Example

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: pytest tests/ --cov=app
    
    - name: Run security scan
      run: bandit -r app/

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Build Docker image
      run: docker build -t gcr.io/project-id/sentinel-x:${{ github.sha }} .
    
    - name: Push to registry
      run: docker push gcr.io/project-id/sentinel-x:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/sentinel-x-api \
          sentinel-x=gcr.io/project-id/sentinel-x:${{ github.sha }} \
          -n sentinel-x
```

---

## Contributing Guidelines

### Code Style

- Follow PEP 8
- Use type hints
- Maximum line length: 100 characters
- Document all public functions

```python
# Example
def create_threat(
    threat_type: str,
    severity: str,
    source_ip: str,
    target_ip: str
) -> Threat:
    """
    Create a new threat record.
    
    Args:
        threat_type: Type of threat (malware, intrusion, etc.)
        severity: Severity level (LOW, MEDIUM, HIGH, CRITICAL)
        source_ip: Source IP address
        target_ip: Target IP address
    
    Returns:
        Threat: Created threat object
    
    Raises:
        ValueError: If severity level is invalid
    """
    ...
```

### Git Workflow

1. Create feature branch: `git checkout -b feature/threat-detection`
2. Make changes and commit: `git commit -m "feat: add threat detection"`
3. Push to remote: `git push origin feature/threat-detection`
4. Create pull request
5. Address review comments
6. Merge when approved

### Commit Messages

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

### Pull Request Process

1. Update documentation
2. Add tests for new features
3. Ensure all tests pass
4. Request review from team members
5. Address feedback
6. Merge when approved

---

## Troubleshooting

### Common Issues

#### Database Connection Error
```
Error: could not connect to server: Connection refused
```

**Solution**:
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Start PostgreSQL
sudo systemctl start postgresql

# Check connection string
echo $DATABASE_URL

# Test connection
psql $DATABASE_URL -c "SELECT 1"
```

#### Redis Connection Error
```
Error: ConnectionError: Error -1 connecting to localhost:6379
```

**Solution**:
```bash
# Check Redis status
redis-cli ping

# Start Redis
redis-server

# Check connection
redis-cli
```

#### JWT Token Expired
```
Error: Token has expired
```

**Solution**:
```bash
# Refresh token via API
POST /api/v1/auth/refresh
Authorization: Bearer <expired_token>

# Or login again
POST /api/v1/auth/login
```

#### Model Loading Error
```
Error: FileNotFoundError: models/anomaly_detection_v1.pkl not found
```

**Solution**:
```bash
# Download models
python scripts/download_models.py

# Or train models
python scripts/train_all_models.py
```

### Debugging

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
uvicorn app.main:app --reload

# Check application logs
tail -f logs/sentinel-x.log

# Database debugging
psql $DATABASE_URL -c "SELECT * FROM threats LIMIT 5;"

# Redis debugging
redis-cli KEYS "*"
redis-cli GET <key>
```

### Performance Issues

```bash
# Check system resources
htop

# Database slow queries
EXPLAIN ANALYZE SELECT ...;

# Redis memory usage
redis-cli INFO memory

# Application profiling
python -m cProfile -s cumulative app/main.py
```

---

## Additional Resources

### Documentation
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)

### Tools
- Postman: API testing
- DBeaver: Database management
- Docker Desktop: Container management
- VS Code: Code editor

### Contacts
- **Lead Developer**: [Contact information]
- **DevOps**: [Contact information]
- **Security**: [Contact information]

---

## License

Sentinel-X is licensed under the MIT License. See LICENSE file for details.

---

**Last Updated**: 2026-01-10  
**Version**: 1.0.0  
**Maintainer**: ShivamGawade-XS
