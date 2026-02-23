# Sentinel-X Backend API

Advanced threat detection and analysis platform backend built with FastAPI.

## Overview

The Sentinel-X backend API provides comprehensive threat detection, analysis, and reporting capabilities. It includes:

- **Threat Detection**: Analyze artifacts (hashes, URLs, IPs) for threat indicators
- **Threat Analysis**: Detailed threat identification and behavior analysis
- **Security Events**: Log and track security events with correlation
- **Statistics & Reporting**: Comprehensive threat metrics and trend analysis
- **RESTful API**: Fully documented OpenAPI/Swagger documentation

## Architecture

```
backend/
├── main.py                 # Main FastAPI application
├── config.py              # Configuration management
├── database.py            # Database connection and utilities
├── models.py              # SQLAlchemy ORM models
├── schemas.py             # Pydantic request/response schemas
├── requirements.txt       # Python dependencies
├── Makefile              # Development commands
│
├── routers/              # API endpoint routers
│   ├── threats.py        # Threat detection endpoints
│   ├── events.py         # Security event endpoints
│   ├── statistics.py     # Analytics and reporting endpoints
│   └── __init__.py      # Router package
│
└── tests/                # Test suite
    ├── conftest.py       # Test configuration and fixtures
    ├── test_api.py       # API endpoint tests
    └── __init__.py
```

## Features

### Threat Detection API
- **POST /api/v1/threats/detect** - Detect threats in artifacts
- **POST /api/v1/threats/analyze** - Perform detailed threat analysis
- **GET /api/v1/threats** - Retrieve threats with filtering
- **GET /api/v1/threats/{threat_id}** - Get specific threat details

### Security Events API
- **POST /api/v1/events/log** - Log security events
- **GET /api/v1/events** - Retrieve events with filtering
- **GET /api/v1/events/{event_id}** - Get specific event details
- **GET /api/v1/events/stats/by-type** - Event statistics by type
- **GET /api/v1/events/stats/by-severity** - Event statistics by severity

### Statistics & Analytics API
- **GET /api/v1/statistics/threats** - Comprehensive threat statistics
- **GET /api/v1/statistics/threats/by-type** - Threat distribution by type
- **GET /api/v1/statistics/threats/by-severity** - Threat distribution by severity
- **GET /api/v1/statistics/system/overview** - System health metrics
- **GET /api/v1/statistics/trends/threat-detection** - Threat detection trends

### Health Check API
- **GET /health** - Service health check
- **GET /api/v1/health** - API v1 health check with database status

## Quick Start

### Prerequisites
- Python 3.9+
- pip or poetry for dependency management

### Installation

1. **Clone the repository**
   ```bash
   cd backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   make install
   ```

4. **Setup environment variables**
   ```bash
   cp .env.example .env
   ```

5. **Initialize database**
   ```bash
   make db-init
   ```

### Development

**Start development server with auto-reload:**
```bash
make run
# Server runs on http://localhost:8000
# API docs available at http://localhost:8000/docs
```

**Run tests:**
```bash
make test
```

**Run tests with coverage:**
```bash
make test-cov
```

**Code formatting and linting:**
```bash
make format    # Auto-format code with black and isort
make lint      # Run flake8 linting checks
make type-check  # Run mypy type checking
```

## API Documentation

When the server is running, visit http://localhost:8000/docs for interactive Swagger UI documentation.

### Example Requests

**Detect Threat:**
```bash
curl -X POST "http://localhost:8000/api/v1/threats/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "sample": "malware_detected_pattern",
    "sample_type": "hash",
    "priority": "HIGH"
  }'
```

**Log Security Event:**
```bash
curl -X POST "http://localhost:8000/api/v1/events/log" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "SUSPICIOUS_LOGIN",
    "severity": "HIGH",
    "source": "192.168.1.100",
    "description": "Multiple failed login attempts detected"
  }'
```

**Get Threat Statistics:**
```bash
curl "http://localhost:8000/api/v1/statistics/threats?days=30"
```

## Database

Sentinel-X uses SQLAlchemy ORM with support for multiple databases:
- SQLite (for development/testing)
- PostgreSQL (for production)
- MySQL (alternative for production)

### Database Models

- **Threat**: Detected threats with classifications
- **ThreatIndicator**: Indicators of compromise (IOCs)
- **AffectedSystem**: Systems impacted by threats
- **ThreatRecommendation**: Recommended remediation actions
- **ThreatAnalysis**: Analysis results and status tracking
- **SecurityEvent**: Raw security events from systems
- **ThreatReport**: Generated threat reports
- **AuditLog**: API access and change tracking

## Configuration

Configuration is managed through:
1. **Environment variables** (.env file)
2. **config.py** - Default and environment-specific settings

Key configuration options:
- `DATABASE_URL` - Database connection string
- `CORS_ORIGINS` - Allowed CORS origins
- `LOG_LEVEL` - Logging verbosity
- `SECRET_KEY` - Application secret key
- `JWT_SECRET_KEY` - JWT signing key

## Error Handling

The API implements comprehensive error handling with:
- HTTP status codes
- Detailed error messages
- Validation error reporting
- Exception logging

## Testing

The test suite includes:
- Health check endpoint tests
- Threat detection endpoint tests
- Threat analysis tests
- Security event logging tests
- Statistics endpoint tests
- Error handling tests

Run tests:
```bash
make test           # Run all tests
make test-cov       # Run with coverage report
make test-watch     # Watch mode (rerun on changes)
```

## Production Deployment

**Using Gunicorn:**
```bash
make run-prod
```

This runs with multiple workers for better performance.

**Using Docker:**
```bash
docker build -t sentinel-x-api .
docker run -p 8000:8000 sentinel-x-api
```

## Performance

- Request/response logging middleware
- GZIP compression for responses
- Database connection pooling
- Efficient pagination for large result sets
- Query optimization with indexed fields

## Security

- Input validation on all requests
- SQL injection prevention (SQLAlchemy ORM)
- CORS middleware configuration
- JWT token support (framework ready)
- Environment variable secrets management

## Logging

Logging is configured with:
- Console output for development
- Rotating file handler for production
- Request/response logging middleware
- Per-module loggers

## Troubleshooting

**Port 8000 already in use:**
```bash
# Use a different port
python -m uvicorn main:app --port 8001
```

**Database connection issues:**
```bash
# Check DATABASE_URL in .env
# For SQLite: sqlite:///./sentinel.db
# For PostgreSQL: postgresql://user:password@localhost/dbname
```

**Import errors:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## Contributing

1. Create a feature branch
2. Make changes with tests
3. Run `make format` and `make lint`
4. Submit pull request

## License

Proprietary - Sentinel-X

## Support

For issues and questions, please contact the development team.
