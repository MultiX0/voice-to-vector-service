# Voice Embedding API - Docker Management

.PHONY: help build up down logs shell test clean

# Default target
help:
	@echo "Voice Embedding API - Available commands:"
	@echo ""
	@echo "  build     - Build the Docker image"
	@echo "  up        - Start all services"
	@echo "  up-simple - Start only the API service"
	@echo "  down      - Stop and remove containers"
	@echo "  logs      - Show logs"
	@echo "  shell     - Open shell in API container"
	@echo "  test      - Test the API endpoints"
	@echo "  clean     - Remove all containers, images, and volumes"
	@echo "  restart   - Restart all services"
	@echo ""

# Build the Docker image
build:
	docker-compose build

# Start all services (API + Redis + Nginx)
up:
	docker-compose up -d
	@echo "Services started!"
	@echo "API: http://localhost:8000"
	@echo "Nginx: http://localhost:80"
	@echo "Docs: http://localhost:8000/docs"

# Start only the API service (minimal)
up-simple:
	docker-compose up -d voice-embedding-api
	@echo "API started at http://localhost:8000"
	@echo "Documentation: http://localhost:8000/docs"

# Stop all services
down:
	docker-compose down

# Show logs
logs:
	docker-compose logs -f

# Show API logs only
logs-api:
	docker-compose logs -f voice-embedding-api

# Open shell in API container
shell:
	docker-compose exec voice-embedding-api /bin/bash

# Restart services
restart:
	docker-compose restart

# Test the API
test:
	@echo "Testing API health..."
	@curl -s http://localhost:8000/health | jq .
	@echo ""
	@echo "Testing root endpoint..."
	@curl -s http://localhost:8000/ | jq .

# Clean everything (careful!)
clean:
	docker-compose down -v --rmi all
	docker system prune -f

# Check service status
status:
	docker-compose ps

# Pull latest base images
pull:
	docker-compose pull

# Rebuild without cache
rebuild:
	docker-compose build --no-cache

# View resource usage
stats:
	docker stats