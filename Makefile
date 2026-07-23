COMPOSE := docker compose
ALL_PROFILES := --profile gpu --profile cpu --profile smoke

# Model for `make smoke`; override per-invocation, e.g. `make smoke MODEL=gpt2-large`
MODEL ?= gpt2
export LENS_MODEL := $(MODEL)

# OFFLINE=1 skips the Hub revision check for already-cached models -- slightly
# faster startup and no "unauthenticated requests" warning. Fails on a model
# that isn't in the cache yet, so leave it off the first time you try one.
OFFLINE ?= 0
export HF_HUB_OFFLINE := $(OFFLINE)

.DEFAULT_GOAL := help
.PHONY: help build up up-cpu web smoke down clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

build: ## Build all images (web, engine, smoke)
	$(COMPOSE) $(ALL_PROFILES) build

up: ## Web + engine (GPU) at http://localhost:5180 (WEB_PORT=... to remap)
	$(COMPOSE) --profile gpu up --build --watch

up-cpu: ## Web + engine (CPU) -- slower J-lens, no GPU needed
	$(COMPOSE) --profile cpu up --build --watch

web: ## Just the Svelte sandbox (every lens except Logit·real)
	$(COMPOSE) up --build --watch web

smoke: ## Bits-ladder sanity check on GPU; PROMPT="..." and MODEL=... to override
	$(COMPOSE) run --rm --build smoke python smoke.py $(if $(PROMPT),"$(PROMPT)",)

down: ## Stop and remove containers
	$(COMPOSE) $(ALL_PROFILES) down

clean: down ## Also remove images built by this project
	$(COMPOSE) $(ALL_PROFILES) down --rmi local
