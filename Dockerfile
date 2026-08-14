# ---- web: the Svelte/Vite sandbox, served by Vite's dev server under bun ----
# Dev server on purpose (not a static build): this is a sandbox, and compose
# watch syncs web/src into the container so HMR keeps working.
FROM oven/bun:1 AS web

WORKDIR /app

COPY web/package.json web/bun.lock ./
RUN bun install --frozen-lockfile

COPY web/index.html web/svelte.config.js web/tsconfig.json web/vite.config.ts ./
COPY web/src ./src/

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

# Triton JIT-compiles CUDA kernels at runtime (the J-lens hits this on some
# architectures, e.g. Qwen) and shells out to a C compiler to build the launcher
# -- the slim base has none, so without this it fails with "Failed to find C
# compiler". libc6-dev must be named explicitly: it's only a Recommends of gcc,
# so --no-install-recommends leaves gcc with no libc headers and every compile
# dies on `stdio.h: No such file`. Keep the layer small and cached before deps.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
RUN --mount=type=bind,source=engine/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=engine/uv.lock,target=uv.lock \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev
ENV PATH="/app/.venv/bin:$PATH"

RUN mkdir -p data

COPY engine/*.py ./

EXPOSE 5181
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5181"]


# ---- tui: the OpenTUI terminal front-end (Bun + a native Zig core) ----
# No CUDA and no model cache in this image; it reaches the engine over HTTP by
# service name. `bun install` pulls @opentui/core's prebuilt native binary for
# this platform as an optional dependency, so no Zig toolchain is needed here.
# Interactive -- run it with `docker compose run --rm tui` (not `up`).
FROM oven/bun:1 AS tui

WORKDIR /app

COPY tui/package.json tui/bun.lock ./
RUN bun install --frozen-lockfile

COPY tui/tsconfig.json ./
COPY tui/src ./src/

CMD ["bun", "run", "src/index.tsx"]
