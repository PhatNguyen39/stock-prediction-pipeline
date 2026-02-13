.PHONY: help install train serve example docker-build docker-up docker-down clean test deploy

help:
	@echo "Stock Prediction Pipeline - Available Commands"
	@echo "================================================"
	@echo "install       - Install dependencies"
	@echo "train         - Train the model"
	@echo "serve         - Start API server"
	@echo "example       - Run simple example"
	@echo "docker-build  - Build Docker image"
	@echo "docker-up     - Start Docker containers"
	@echo "docker-down   - Stop Docker containers"
	@echo "clean         - Clean generated files"
	@echo "test          - Run tests"
	@echo "deploy        - Deploy to Fly.io"

install:
	pip install -r requirements.txt
	cp .env.example .env
	@echo "Installation complete! Edit .env file with your configuration."

train:
	python train.py

serve:
	python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

example:
	python example.py

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d
	@echo "API running at http://localhost:8000"
	@echo "Check logs: docker-compose logs -f"

docker-down:
	docker-compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage

test:
	pytest tests/ -v

deploy:
	fly deploy
