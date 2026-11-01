FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[api]"

# Provider and key come from environment variables at run time, never baked in:
#   docker run -p 8000:8000 --env-file .env screening-classifier
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "screener.api:app", "--host", "0.0.0.0", "--port", "8000"]
