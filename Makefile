.PHONY: install train evaluate lint test run clean

install:
	@echo "Creating virtual environment..."
	@python3 -m venv .venv
	@echo "Installing requirements..."
	@.venv/bin/pip install -q -r requirements.txt
	@echo "Done! Activate with: source .venv/bin/activate"

train:
	@echo "Training model..."
	@cd src && ../.venv/bin/python training/train.py

evaluate:
	@echo "Evaluating model..."
	@cd src && ../.venv/bin/python training/evaluate.py

lint:
	@echo "Linting..."
	@.venv/bin/ruff check src/ tests/

test:
	@echo "Running tests..."
	@.venv/bin/pytest tests/ -v

run:
	@echo "Starting API at http://localhost:8000"
	@cd src && ../.venv/bin/uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

mlflow-ui:
	@echo "Starting MLflow UI at http://localhost:5000"
	@.venv/bin/mlflow ui --backend-store-uri sqlite:///notebooks/mlflow.db --port 5000

clean:
	@rm -rf .venv __pycache__ src/__pycache__ .pytest_cache .ruff_cache
	@find . -name "*.pyc" -delete
	@echo "Cleaned."
