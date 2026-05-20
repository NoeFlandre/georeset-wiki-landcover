FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --group dev --no-install-project

COPY LICENSE ./
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests
RUN uv sync --frozen --group dev

CMD ["uv", "run", "python", "-c", "print('GeoReset Wiki Land-Cover container ready. Mount ./data at /app/data and run a documented module command.')"]
