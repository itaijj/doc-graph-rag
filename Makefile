.PHONY: up down init-db ingest ask test lint

up:
	docker compose up -d

down:
	docker compose down

init-db:
	python -m app.db.init

ingest:
	python scripts/ingest_folder.py $(FOLDER)

ask:
	python scripts/ask.py "$(Q)"

test:
	pytest tests/ -v

lint:
	ruff check app scripts tests
