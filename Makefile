.PHONY: install dev test lint format typecheck check run clean

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements.txt

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy app

check:
	ruff check .
	ruff format --check .
	pytest

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

benchmark:
	python scripts/benchmark.py

benchmark-concurrent:
	python scripts/benchmark_concurrent.py

#1.ruff check . --fix
#2.ruff format .
#3.make check