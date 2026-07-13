.PHONY: setup index test eval api docker-build docker-up drift clean

setup:
	pip install -r requirements.txt

index:
	cd src && python indexer.py

test:
	PYTHONPATH=. pytest tests/ -v

eval:
	python eval/evaluate.py

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -f docker/Dockerfile -t llm-support-api:latest .

docker-up:
	docker compose -f docker/docker-compose.yml up --build

pipeline: index test eval
	@echo "Pipeline completo: indexação -> testes -> avaliação. OK."

clean:
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
