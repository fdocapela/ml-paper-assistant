.PHONY: setup run test down build logs shell ingest cache

ENV_FILE := .env

# ── Prerequisites ─────────────────────────────────────────────────────────────
check-env:
	@test -f $(ENV_FILE) || \
	  (echo "❌ .env not found. Run: cp .env.example .env then add your GOOGLE_API_KEY" && exit 1)
	@grep -q "your_google_api_key_here" $(ENV_FILE) && \
	  (echo "❌ Please set your GOOGLE_API_KEY in .env" && exit 1) || true

# ── Cache: download packages to pip_cache/ (host must have internet) ──────────
cache:
	@echo "📦 Downloading packages for Linux x86_64..."
	@mkdir -p pip_cache
	@pip3 download \
	  --dest pip_cache \
	  --timeout 120 \
	  --platform manylinux_2_17_x86_64 \
	  --platform manylinux2014_x86_64 \
	  --platform linux_x86_64 \
	  --python-version 311 \
	  --implementation cp \
	  --abi cp311 \
	  --only-binary=:all: \
	  -r requirements.txt 2>/dev/null || true
	@pip3 download --dest pip_cache --timeout 120 -r requirements.txt
	@echo "✅ Cache ready: $$(ls pip_cache | wc -l) packages"

# ── Setup: single command that does everything ────────────────────────────────
setup: check-env
	@if [ ! -d pip_cache ] || [ "$$(ls pip_cache 2>/dev/null | wc -l)" -lt 50 ]; then \
	  echo "📦 pip_cache/ not found — downloading packages first..."; \
	  $(MAKE) cache; \
	fi
	@echo "🐳 Building Docker image..."
	@docker compose build
	@echo "🚀 Starting containers..."
	@docker compose up -d
	@echo "⏳ Waiting for ChromaDB to be ready..."
	@sleep 20
	@echo "🔢 Ingesting papers into ChromaDB..."
	@docker compose exec app python3 scripts/ingest_papers.py
	@echo ""
	@echo "✅ Setup complete!"
	@echo "   UI:      http://localhost:8000"
	@echo "   API:     http://localhost:8000/api/docs"

# ── Run: execute the 5 evaluation questions via API ──────────────────────────
run:
	@echo "🚀 Running evaluation questions..."
	@docker compose exec app python3 scripts/run_evaluation.py

# ── Test ──────────────────────────────────────────────────────────────────────
test:
	@echo "🧪 Running tests..."
	@docker compose exec app pytest tests/ -v --tb=short

# ── Down ──────────────────────────────────────────────────────────────────────
down:
	@echo "🛑 Shutting down..."
	@docker compose down -v
	@echo "Done."

# ── Helpers ───────────────────────────────────────────────────────────────────
build:
	@docker compose build

logs:
	@docker compose logs -f app

shell:
	@docker compose exec app bash

ingest:
	@docker compose exec app python3 scripts/ingest_papers.py --reset
