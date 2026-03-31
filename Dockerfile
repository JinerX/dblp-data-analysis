FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry

COPY pyproject.toml poetry.lock* ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token='nokia'"]
