.PHONY: test test-cov backend-install frontend-install eval-retrieval eval-generation eval-dry-run up down

backend-install:
	cd backend && pip install -e ".[dev]"

frontend-install:
	cd frontend && npm install

test:
	cd backend && python -m pytest tests/ -v

test-cov:
	cd backend && python -m pytest tests/ --cov=app --cov-report=term-missing

eval-dry-run:
	cd eval && python run_retrieval_eval.py --dry-run && python run_generation_eval.py --dry-run

eval-retrieval:
	cd eval && python run_retrieval_eval.py

eval-generation:
	cd eval && python run_generation_eval.py

up:
	docker compose up --build

down:
	docker compose down
