.PHONY: help install dev test lint format clean

help:
	@echo "Sentinel-X Backend Development Commands"
	@echo "========================================"
	@echo ""
	@echo "Setup:"
	@echo "  make install      Install dependencies"
	@echo "  make dev          Install in development mode"
	@echo ""
	@echo "Running:"
	@echo "  make run          Run development server"
	@echo "  make run-prod     Run production server with gunicorn"
	@echo ""
	@echo "Testing:"
	@echo "  make test         Run test suite"
	@echo "  make test-cov     Run tests with coverage report"
	@echo "  make test-watch   Watch tests (rerun on file changes)"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint         Run linting checks (flake8, pylint)"
	@echo "  make format       Format code with black and isort"
	@echo "  make type-check   Run type checking with mypy"
	@echo ""
	@echo "Database:"
	@echo "  make db-init      Initialize database"
	@echo "  make db-reset     Reset database (drops all tables)"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean        Remove cache and temp files"
	@echo "  make docs         Generate API documentation"

install:
	pip install -q --upgrade pip
	pip install -r requirements.txt

dev:
	pip install -q --upgrade pip
	pip install -r requirements.txt
	pip install pytest pytest-asyncio pytest-cov black flake8 mypy isort

run:
	@echo "Starting development server on http://localhost:8000"
	@echo "API docs available at http://localhost:8000/docs"
	python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

run-prod:
	gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term

test-watch:
	pytest-watch tests/ -- -v

lint:
	flake8 . --count --show-source --statistics --exclude=./venv,./env,./build
	pylint **/*.py --fail-under=7.0 2>/dev/null || echo "Note: Pylint checks completed"

format:
	black . --exclude="/(\.venv|venv|\.env|build|\.git)/"
	isort . --skip="venv,\.venv,.env,build"

type-check:
	mypy . --ignore-missing-imports

db-init:
	python -c "from database import init_db; init_db(); print('Database initialized!')"

db-reset:
	python -c "from database import reset_db; reset_db(); print('Database reset!')"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/ dist/ build/ *.egg-info/

docs:
	@echo "API documentation is available at: http://localhost:8000/docs (when server is running)"
	@echo "OpenAPI schema: http://localhost:8000/openapi.json"

# Install pre-commit hooks for automated checks
setup-hooks:
	pip install pre-commit
	pre-commit install
	@echo "Pre-commit hooks installed!"
