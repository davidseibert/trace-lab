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


# ---- engine: the real-model lens service on the torch CUDA base ----
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime AS engine

ENV TZ=America/New_York
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# uv instead of pip: grab the static binary from the official image and install
# the requirements into the base image's existing (torch-equipped) environment.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
RUN --mount=type=bind,source=engine/requirements.txt,target=/tmp/requirements.txt \
    uv pip install --system --no-cache --requirement /tmp/requirements.txt

RUN mkdir -p data

COPY engine/lens.py engine/main.py engine/smoke.py ./

EXPOSE 5181
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5181"]
