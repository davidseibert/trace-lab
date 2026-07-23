# ---- web: the Svelte/Vite sandbox, served by Vite's dev server under bun ----
# Dev server on purpose (not a static build): this is a sandbox, and compose
# watch syncs src/ into the container so HMR keeps working.
FROM oven/bun:1 AS web

WORKDIR /app

COPY package.json bun.lock ./
RUN bun install --frozen-lockfile

COPY index.html svelte.config.js tsconfig.json vite.config.ts ./
COPY src ./src/

EXPOSE 5180
CMD ["bun", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5180", "--strictPort"]


# ---- engine: the real-model lens service ----
# One dependency story everywhere: the same pyproject.toml + uv.lock that drive
# a local `uv sync` build this image too. The lockfile already pins CUDA torch
# for linux (cu126 wheels bundle the CUDA libs), so a plain python base
# suffices -- no pytorch base image, no separate requirements file.
FROM python:3.12-slim AS engine

ENV TZ=America/New_York
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
RUN --mount=type=bind,source=engine/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=engine/uv.lock,target=uv.lock \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev
ENV PATH="/app/.venv/bin:$PATH"

RUN mkdir -p data

COPY engine/lens.py engine/main.py engine/smoke.py engine/train.py ./

EXPOSE 5181
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5181"]
