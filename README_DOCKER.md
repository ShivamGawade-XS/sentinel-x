# Sentinel-X Docker Compose Setup

## Quick Start

### Prerequisites
- Docker & Docker Compose installed
- Git installed

### 1. Clone and Setup

```bash
git clone https://github.com/ShivamGawade-XS/sentinel-x.git
cd sentinel-x
cp .env.example .env
```

### 2. Build and Run

```bash
# Development mode (default, includes hot-reload for both frontend and backend)
docker-compose up -d

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Stop services
docker-compose down
```

### 3. Access Services

- **Frontend**: http://localhost:3000
- **API Backend**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Project Structure

```
sentinel-x/
├── backend/              # FastAPI application
│   ├── main.py          # App entry point
│   ├── config.py        # Configuration
│   ├── database.py      # DB setup
│   ├── models.py        # SQLAlchemy models
│   ├── schemas.py       # Pydantic schemas
│   ├── routers/         # API endpoints
│   ├── requirements.txt  # Python dependencies
│   └── .env            # Backend config (auto-created)
├── frontend/             # Next.js application
│   ├── app/            # App router
│   ├── package.json    # Node dependencies
│   └── .env.local      # Frontend config (auto-created)
├── nginx/               # Nginx configuration
│   ├── nginx.conf      # Main config
│   └── conf.d/         # Site configs
├── Dockerfile           # Backend container
├── Dockerfile.frontend  # Frontend container
├── docker-compose.yml   # Main compose file
├── docker-compose.override.yml  # Dev overrides
└── .env.example        # Environment template
```

## Service Details

### Backend (FastAPI)
- **Port**: 8000
- **Hot Reload**: Enabled in dev mode
- **Database**: PostgreSQL (auto-initialized)
- **Cache**: Redis

### Frontend (Next.js)
- **Port**: 3000
- **Hot Reload**: Enabled in dev mode
- **API URL**: Points to backend on localhost:8000

### Database (PostgreSQL)
- **Port**: 5432 (exposed for local development)
- **Auto-init**: Database and tables created on startup

### Cache (Redis)
- **Port**: 6379 (exposed for development)

### Reverse Proxy (Nginx)
- **Port**: 80 (prod profile only)
- **Status**: Optional (not started by default)

## Common Commands

### Start Services
```bash
# Start all services (development mode)
docker-compose up -d

# Start specific service
docker-compose up -d backend
docker-compose up -d frontend

# Rebuild images
docker-compose up -d --build
```

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend

# Last 100 lines
docker-compose logs --tail=100 backend
```

### Execute Commands
```bash
# Run command in backend
docker-compose exec backend python -c "from database import init_db; init_db()"

# Access backend shell
docker-compose exec backend bash

# Access frontend shell
docker-compose exec frontend sh
```

### Database Management
```bash
# Access PostgreSQL
docker-compose exec postgres psql -U postgres -d sentinel_x

# Reset database
docker-compose exec backend python -c "from database import reset_db; reset_db()"
```

### Stop and Cleanup
```bash
# Stop services
docker-compose stop

# Stop and remove containers
docker-compose down

# Remove volumes too
docker-compose down -v

# Clean everything
docker-compose down -v --remove-orphans
```

## Environment Variables

Edit `.env` to customize:

```env
# Ports
BACKEND_PORT=8000
FRONTEND_PORT=3000
POSTGRES_PORT=5432

# Database
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=sentinel_x

# Security (change in production!)
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-here
```

## Troubleshooting

### Backend won't start
```bash
# Check logs
docker-compose logs -f backend

# Reinitialize database
docker-compose exec backend python -c "from database import init_db; init_db()"

# Rebuild image
docker-compose up -d --build backend
```

### Frontend can't connect to backend
- Ensure `NEXT_PUBLIC_API_URL` in `.env.local` is correct
- Check that backend is running: `docker-compose logs backend`
- Verify CORS settings in backend config

### Database connection fails
```bash
# Check PostgreSQL logs
docker-compose logs postgres

# Verify connection
docker-compose exec postgres pg_isready
```

### Redis connection issues
```bash
# Test Redis
docker-compose exec redis redis-cli ping
```

## Production Deployment

For production:

1. Update `.env` with production values
2. Use Nginx profile:
   ```bash
   docker-compose --profile prod up -d
   ```
3. Set up SSL certificates in `./nginx/ssl/`
4. Configure domains in Nginx config
5. Use production-grade reverse proxy settings

## Development Workflow

### Making Code Changes

**Backend**: Changes auto-reload via Uvicorn
```python
# Edit backend/routers/threats.py
# Changes auto-apply on save
```

**Frontend**: Changes auto-reload via Next.js
```tsx
// Edit frontend/app/page.tsx
// Changes auto-apply on save
```

### Running Tests

```bash
# Backend tests
docker-compose exec backend pytest tests/ -v

# Backend coverage
docker-compose exec backend pytest tests/ --cov=.

# Frontend tests
docker-compose exec frontend npm test
```

### Linting & Formatting

```bash
# Backend
docker-compose exec backend black .
docker-compose exec backend flake8 .

# Frontend
docker-compose exec frontend npm run lint:fix
```

## Monitoring & Optional Services

### Start Monitoring Stack (Prometheus + Grafana)
```bash
docker-compose --profile monitoring up -d
```

Access:
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3001 (admin/admin)

## Notes

- Database is initialized automatically on first run
- Both backend and frontend have hot-reload enabled in dev mode
- All services are networked on `sentinel-network`
- Volumes persist data between restarts
- `.env` file is gitignored for security
