.PHONY: help, install-deps, lint, test, coverage
.DEFAULT_GOAL := help

help: ## display this help section
	@awk 'BEGIN {FS = ":.*?## "}; /^[a-zA-Z_-]+:.*?##/ {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install-deps: ## install the package-s dev dependencies
	python -m pip install -r requirements-dev.txt

lint: ## check style with flake8
	tox -e lint

test: ## run tests on every supported Python version
	tox

coverage: ## check code coverage
	tox -e coverage


