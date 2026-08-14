COMPOSE := docker compose
ALL_PROFILES := --profile gpu --profile cpu --profile tui --profile smoke

# Exported to EVERY target as LENS_MODEL: sets the engine's default model for
# `make up`/`make tui` as well as the model `make smoke` checks. Override
# per-invocation, e.g. `make smoke MODEL=gpt2-large`.
MODEL ?= gpt2
export LENS_MODEL := $(MODEL)

# OFFLINE=1 skips the Hub revision check for already-cached models -- slightly
# faster startup and no "unauthenticated requests" warning. Fails on a model
# that isn't in the cache yet, so leave it off the first time you try one.
OFFLINE ?= 0
export HF_HUB_OFFLINE := $(OFFLINE)

.DEFAULT_GOAL := help
.PHONY: help build up up-cpu web tui smoke train down clean volume

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

# The shared model volume is `external` in compose (so `down -v` can never
# delete it); this idempotent create covers fresh machines.
volume:
	@docker volume create hf-cache >/dev/null

build: ## Build all images (web, engine, smoke)
	$(COMPOSE) $(ALL_PROFILES) build

up: volume ## Web + engine (GPU) at http://localhost:5180 (WEB_PORT=... to remap)
	$(COMPOSE) --profile gpu up --build --watch

up-cpu: volume ## Web + engine (CPU) -- slower J-lens, no GPU needed
	$(COMPOSE) --profile cpu up --build --watch

web: ## Just the Svelte sandbox (every lens except Logit·real)
	$(COMPOSE) up --build --watch web

# The engine must be listening before the TUI attaches; the first line's --wait
# blocks on its healthcheck. Deliberately no --build on that line: reuse whatever
# `make up`/`make build` already started -- rebuilding and re-exporting the
# multi-GB torch image on every `make tui` is slow and pointless. If the engine
# isn't up yet, `up` builds it once; after that this is a fast no-op. The second
# line builds and runs only the small TUI image and connects over the network.
# LENS_ENGINE=http://engine-cpu:5181 (with a running engine-cpu) points the TUI
# at a CPU engine instead.
# --service-ports publishes the spectate sidecar (5182) that the MCP bridge
# reads; `compose run` skips the service's port mappings without it.
tui: volume ## Launch the OpenTUI terminal front-end (GPU engine starts first)
	$(COMPOSE) --profile gpu up -d --wait engine-gpu
	$(COMPOSE) run --rm --build --service-ports tui

smoke: volume ## Bits-ladder sanity check on GPU; PROMPT="..." and MODEL=... to override
	$(COMPOSE) run --rm --build smoke python smoke.py $(if $(PROMPT),"$(PROMPT)",)

# Trains into the shared hf-cache volume; checkpoints appear in the model
# dropdown as local/add-step0, local/add-mid, local/add-final.
EPOCHS ?= 40
train: volume ## Train the 2-digit addition GPT-2 (GPU); EPOCHS=... to override
	$(COMPOSE) run --rm --build -e EPOCHS=$(EPOCHS) smoke python train.py

down: ## Stop and remove containers
	$(COMPOSE) $(ALL_PROFILES) down

clean: down ## Also remove images built by this project
	$(COMPOSE) $(ALL_PROFILES) down --rmi local
