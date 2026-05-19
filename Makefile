HOST ?= 127.0.0.1
BACKEND_PORT ?= 8010
FRONTEND_PORT ?= 5174
PYTHON ?= python3
VENV_BIN := .venv/bin
PIP := $(VENV_BIN)/pip
PYTEST := $(VENV_BIN)/pytest

.PHONY: help venv install frontend-install test backend frontend dev clean

help:
	@echo "TripMind commands:"
	@echo "  make venv              Create the Python virtual environment"
	@echo "  make install           Install Python dependencies into .venv"
	@echo "  make frontend-install  Install frontend dependencies"
	@echo "  make test              Run backend tests"
	@echo "  make backend           Start the FastAPI server"
	@echo "  make frontend          Start the Vite frontend"
	@echo "  make dev               Start backend and frontend together"
	@echo "  make clean             Remove caches"

venv:
	$(PYTHON) -m venv .venv

install: venv
	$(PIP) install -e '.[dev]'

frontend-install:
	cd frontend && npm install

test:
	$(PYTEST) -q

backend:
	$(VENV_BIN)/python -m uvicorn tripmind.api:app --host $(HOST) --port $(BACKEND_PORT)

frontend:
	cd frontend && npm run dev -- --host $(HOST) --port $(FRONTEND_PORT)

dev:
	@trap 'kill 0' INT TERM EXIT; \
	$(MAKE) backend & \
	$(MAKE) frontend & \
	wait

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name ".pytest_cache" -type d -prune -exec rm -rf {} +
