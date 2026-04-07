.PHONY: up down init-db notebook

up:
	docker compose up -d --build

down:
	docker compose down

init-db:
	docker compose exec notebook python -m src.data_loading.alias_builder
	docker compose exec notebook python -m src.data_loading.data_loader

notebook:
	docker compose exec notebook jupyter lab
