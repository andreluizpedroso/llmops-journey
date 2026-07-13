# Etapa 11 — Docker
FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.12-slim
WORKDIR /app

RUN useradd --create-home appuser
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

COPY api/ ./api/
COPY src/ ./src/
COPY guardrails/ ./guardrails/
COPY cache/ ./cache/
COPY monitoring/ ./monitoring/
COPY data/ ./data/

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# ANTHROPIC_API_KEY é opcional: sem ela, a API usa o provider mock local
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
