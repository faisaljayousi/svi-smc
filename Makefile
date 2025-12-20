.PHONY: install run test clean build

install:
	poetry install

run:
	poetry run python main.py

test:
	poetry run pytest

clean:
	rm -rf `find . -name __pycache__`
	rm -f .coverage
	rm -rf .pytest_cache

build:
	docker build -t svi-particle-filter .

format:
	poetry run ruff format .  
	poetry run ruff check . --fix 

lint:
	poetry run ruff check .
