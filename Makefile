# DALO-CP project Makefile.
# The project path contains spaces: every path below is quoted, and no target
# depends on an absolute path.

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

GPU ?= auto
ifeq ($(GPU),auto)
  GPU := $(shell command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 && echo 1 || echo 0)
endif
COMPOSE_FILES := -f docker/compose.yaml
ifeq ($(GPU),1)
  COMPOSE_FILES += -f docker/compose.gpu.yaml
endif
COMPOSE := docker compose --project-directory . $(COMPOSE_FILES) --env-file .env
RUN := $(COMPOSE) run --rm runner python -m delayed_local_cp.cli

EXP ?= configs/experiments/smoke.yaml
JOBS ?= 4

.PHONY: help env build up down logs test smoke experiments resume status figures paper prepare analyze report release shell

help: ## Show targets
	@grep -E '^[a-z-]+:.*## ' Makefile | sed 's/:.*## /\t/' | column -t -s $$'\t'

.env:
	@printf 'HOST_UID=%s\nHOST_GID=%s\nDATASET_DIR="%s"\nDASHBOARD_PORT=8501\n' \
	  "$$(id -u)" "$$(id -g)" "$$(cd ../datasets && pwd -P)" > .env
	@echo "wrote .env"

env: .env ## Create .env from the host (UID/GID, dataset folder)

build: .env ## Build the Docker image
	$(COMPOSE) build runner

up: .env ## Start the dashboard on http://127.0.0.1:8501
	$(COMPOSE) up -d --build dashboard
	@echo "Dashboard: http://127.0.0.1:$${DASHBOARD_PORT:-8501}"

down: .env ## Stop all services and running experiments (runs checkpoint on SIGTERM)
	@ids="$$(docker ps -q --filter label=com.docker.compose.project=dalo-cp --filter label=com.docker.compose.oneoff=True)"; \
	  if [ -n "$$ids" ]; then echo "stopping running experiment containers (SIGTERM, 60s grace)"; docker stop -t 60 $$ids >/dev/null; fi
	$(COMPOSE) --profile gpu down --timeout 60

logs: .env ## Follow dashboard logs
	$(COMPOSE) logs -f dashboard

test: .env ## Run the test suite
	$(COMPOSE) run --rm tests

smoke: .env ## Small end-to-end run on synthetic data
	$(RUN) smoke

experiments: .env ## Run a sweep: make experiments EXP=configs/experiments/<name>.yaml JOBS=4
	$(RUN) sweep --config "$(EXP)" --jobs $(JOBS)

resume: .env ## Report state and restart interrupted runs
	$(RUN) resume --jobs $(JOBS)

status: .env ## Show run registry summary
	$(RUN) status

figures: .env ## Generate figures from results
	$(RUN) figures

paper: ## Build paper/main.pdf with the host LaTeX installation
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

prepare: .env ## Build a processed dataset: make prepare NAME=elec2
	$(RUN) prepare $(NAME)

analyze: .env ## Milestone analysis: make analyze NAME=m2
	$(RUN) analyze $(NAME)

report: .env ## Write a progress report to reports/progress/
	$(RUN) report

release: .env ## Build, check and test github_repo/, then commit locally. Never pushes.
	@test -z "$$(git status --porcelain)" || { echo "release needs a clean working tree"; exit 1; }
	@tag="$$(git describe --tags --exact-match 2>/dev/null)" || { echo "release needs a tag on this commit"; exit 1; }; \
	  test -d github_repo/.git || git clone https://github.com/malekpouri/dalo-cp github_repo; \
	  commit="$$(git rev-parse HEAD)"; \
	  $(COMPOSE) run --rm -e RELEASE_COMMIT="$$commit" -e RELEASE_TAG="$$tag" runner python -m delayed_local_cp.cli release; \
	  rel="docker compose -p dalo-cp-release --project-directory github_repo -f github_repo/docker/compose.yaml --env-file .env"; \
	  $$rel run --rm --build tests; \
	  $$rel run --rm runner python -m delayed_local_cp.cli smoke; \
	  rm -rf github_repo/runs github_repo/reports github_repo/paper; \
	  git -C github_repo add -A; \
	  git -C github_repo -c user.name="$$(git config user.name)" -c user.email="$$(git config user.email)" \
	    commit -m "Release from $$commit ($$tag)"; \
	  echo "github_repo/ committed locally. Pushing needs author approval."

shell: .env ## Open a shell in the runner container
	$(COMPOSE) run --rm runner bash
