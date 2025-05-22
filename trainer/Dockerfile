FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_INSTALL_DIR=/python

RUN uv python install 3.12

# Install project dependencies and the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Uninstall ray
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip uninstall ray

# Use the final image without uv
FROM nvidia/cuda:12.6.3-base-ubuntu22.04

# Copy the application from the builder
COPY --from=builder --chown=python:python /python /python
COPY --from=builder --chown=app:app /app /app

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["python", "/app/src/trainer/main.py"]