-include .env

export

.PHONY: install train evaluate lint test run clean

install:
	@echo "Creating virtual environment..."
	@python3 -m venv .venv
	@echo "Upgrading build tools..."
	@.venv/bin/pip install -q --upgrade pip setuptools wheel
	@echo "Installing requirements..."
	@.venv/bin/pip install -q -e .
	@echo "Configuring nbstripout (strip notebook outputs on commit)..."
	@.venv/bin/nbstripout --install
	@echo "Done! Activate with: source .venv/bin/activate"

train:
	@echo "Fetching data..."
	@curl -L -o data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv   https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
	@echo "Training model..."
	@.venv/bin/python src/ml/train.py

evaluate:
	@echo "Fetching data..."
	@curl -L -o data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
	@echo "Evaluating model..."
	@.venv/bin/python src/ml/evaluate.py

lint:
	@echo "Linting..."
	@.venv/bin/ruff check src/ tests/

format:
	@echo "Linting..."
	@.venv/bin/ruff format --check src/ tests/

test:
	@echo "Running tests..."
	@.venv/bin/pytest tests/unit --cov=src --cov-report=term-missing --cov-report=xml --cov-fail-under=80 -v

e2e:
	@echo "Running tests..."
	@.venv/bin/pytest tests/e2e -v

run:
	@echo "Starting API at http://localhost:8000"
	@.venv/bin/uvicorn app.main:app --reload --app-dir src

mlflow-ui:
	@echo "Starting MLflow UI at http://localhost:5001"
	@.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

plan-aws:
	@echo "Setting up AWS infrastructure..."
	@cd src/infra && terraform init
	@cd src/infra && terraform plan -out=tfplan
	@echo "Review the plan above, then apply with make apply-aws-infra"

build-aws:
	@echo "Deploying AWS infrastructure..."
	@cd src/infra && terraform apply -auto-approve

destroy-aws:
	@echo "Destroying AWS infrastructure..."
	@cd src/infra && terraform destroy

clean:
	@rm -rf .venv __pycache__ src/__pycache__ .pytest_cache .ruff_cache mlflow.db
	@find . -name "*.pyc" -delete
	@echo "Cleaned."
