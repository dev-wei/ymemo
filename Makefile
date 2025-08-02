# YMemo Development Makefile
# Provides convenient commands for common development tasks

.PHONY: help install install-dev test test-coverage test-fast lint format type-check security clean run setup-dev ci-test all-checks

# Default target
help: ## Show this help message
	@echo "YMemo Development Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Environment setup
install: ## Install production dependencies
	source .venv/bin/activate && pip install -r requirements.txt

install-dev: ## Install development dependencies
	source .venv/bin/activate && pip install -e ".[dev]"

setup-dev: install-dev ## Set up complete development environment
	source .venv/bin/activate && pre-commit install
	@echo "✅ Development environment ready!"
	@echo "📝 Try 'make test' to run tests or 'make run' to start the app"

# Testing
test: ## Run full test suite
	source .venv/bin/activate && python -m pytest tests/providers/ tests/aws/ tests/audio/ tests/unit/test_enhanced_session_manager.py tests/unit/test_session_manager_stop.py tests/config/ -v

test-fast: ## Run essential tests only
	source .venv/bin/activate && python -m pytest tests/providers/test_provider_factory.py tests/unit/test_enhanced_session_manager.py tests/config/test_audio_config_validation.py -v

test-coverage: ## Run tests with coverage report
	source .venv/bin/activate && python -m pytest tests/providers/ tests/aws/ tests/audio/ tests/unit/test_enhanced_session_manager.py tests/unit/test_session_manager_stop.py tests/config/ --cov=src --cov-report=html --cov-report=term-missing -v

ci-test: ## Run tests exactly like CI does
	source .venv/bin/activate && python -m pytest tests/providers/ tests/aws/ tests/audio/ tests/unit/test_enhanced_session_manager.py tests/unit/test_session_manager_stop.py tests/config/ --cov=src --cov-report=xml --cov-report=html --cov-report=term-missing --junitxml=pytest-results.xml -v --tb=short --durations=10 --maxfail=5

# Code quality
lint: ## Run linting (ruff)
	source .venv/bin/activate && ruff check src/ tests/ --output-format=github

format: ## Format code (ruff + black + isort)
	source .venv/bin/activate && ruff format src/ tests/
	source .venv/bin/activate && black src/ tests/
	source .venv/bin/activate && isort src/ tests/

type-check: ## Run type checking (mypy)
	source .venv/bin/activate && mypy src/ --ignore-missing-imports --no-strict-optional

security: ## Run security scan (bandit)
	source .venv/bin/activate && bandit -r src/ -f json -o bandit-report.json || true
	@echo "Security scan complete. Check bandit-report.json for details."

# Combined checks
all-checks: lint type-check security ## Run all quality checks
	@echo "✅ All quality checks completed"

pre-commit: ## Run pre-commit hooks on all files
	source .venv/bin/activate && pre-commit run --all-files

# Application
run: ## Start YMemo application
	@echo "🎙️ Starting YMemo..."
	@echo "⚠️  Note: This will start a web interface. Use Ctrl+C to stop."
	source .venv/bin/activate && python main.py

create-test-audio: ## Create test audio file for testing
	source .venv/bin/activate && python tests/create_test_audio.py

# Development utilities
clean: ## Clean up build artifacts and cache
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf build/ dist/ coverage_reports/ htmlcov/ .coverage pytest-results.xml bandit-report.json

# Quick development workflow
dev: format lint test-fast ## Format, lint, and run essential tests
	@echo "✅ Quick development checks passed!"

# CI simulation
ci: clean install-dev all-checks ci-test ## Simulate complete CI pipeline locally
	@echo "🎉 CI simulation completed successfully!"
