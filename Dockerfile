FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir uv && uv sync --frozen || uv sync

COPY . .

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

EXPOSE 8501

CMD ["uv", "run", "python", "main.py", "serve"]
