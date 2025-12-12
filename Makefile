# ============================================
# Digital Human Clone Dashboard - Makefile
# ============================================
#
# Usage: make <target>
# Run 'make help' for available commands
#
# ============================================

# Configuration
COMPOSE_FILE := docker-compose.yml
CONTAINER_NAME := digital-human-dashboard
IMAGE_NAME := voice_model_training-digital-human

# Default target
.DEFAULT_GOAL := help

# ============================================
# BUILD & RUN
# ============================================

.PHONY: build
build: ## Build the Docker image
	docker-compose -f $(COMPOSE_FILE) build

.PHONY: up
up: ## Start container in detached mode
	docker-compose -f $(COMPOSE_FILE) up -d

.PHONY: down
down: ## Stop and remove container
	docker-compose -f $(COMPOSE_FILE) down

.PHONY: restart
restart: down up ## Restart the container

.PHONY: run
run: ## Start container in foreground (with logs)
	docker-compose -f $(COMPOSE_FILE) up

# ============================================
# LOGS & MONITORING
# ============================================

.PHONY: logs
logs: ## Stream container logs (Ctrl+C to exit)
	docker-compose -f $(COMPOSE_FILE) logs -f

.PHONY: status
status: ## Show container status
	docker-compose -f $(COMPOSE_FILE) ps

.PHONY: health
health: ## Check container health status
	@docker inspect --format='{{.State.Health.Status}}' $(CONTAINER_NAME) 2>/dev/null || echo "Container not running"

# ============================================
# SHELL ACCESS
# ============================================

.PHONY: shell
shell: ## Open bash shell in running container
	docker exec -it $(CONTAINER_NAME) bash

.PHONY: python
python: ## Open Python REPL in running container
	docker exec -it $(CONTAINER_NAME) python

# ============================================
# GPU & SYSTEM
# ============================================

.PHONY: gpu
gpu: ## Check GPU availability on host
	nvidia-smi

.PHONY: gpu-container
gpu-container: ## Check GPU inside container
	docker exec -it $(CONTAINER_NAME) nvidia-smi

# ============================================
# CLEANUP
# ============================================

.PHONY: clean
clean: ## Stop container and remove volumes
	docker-compose -f $(COMPOSE_FILE) down -v

.PHONY: clean-all
clean-all: clean ## Full cleanup: containers, images, and build cache
	docker image rm $(IMAGE_NAME) 2>/dev/null || true
	docker builder prune -f

.PHONY: prune
prune: ## Remove all unused Docker resources (use with caution)
	docker system prune -af

# ============================================
# BUILD OPTIONS
# ============================================

.PHONY: rebuild
rebuild: ## Force rebuild without cache
	docker-compose -f $(COMPOSE_FILE) build --no-cache

.PHONY: pull
pull: ## Pull latest base images
	docker-compose -f $(COMPOSE_FILE) pull

# ============================================
# HELP
# ============================================

.PHONY: help
help: ## Show this help message
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║       Digital Human Clone Dashboard - Make Commands        ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Examples:"
	@echo "  make build        # Build the Docker image"
	@echo "  make up           # Start the application"
	@echo "  make logs         # View live logs"
	@echo "  make shell        # Access container shell"
	@echo ""
