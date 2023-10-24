.PHONY: help, clean clean-build clean-pyc clean-test docs
.DEFAULT_GOAL := help

help: ## display this help section
	@awk 'BEGIN {FS = ":.*?## "}; /^[a-zA-Z_-]+:.*?##/ {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

install-deps: clean ## install the package-s dev dependencies
	python -m pip install -r requirements-dev.txt

lint: ## check style with flake8
	tox -e lint

test: ## run tests on every supported Python version
	tox

coverage: ## check code coverage
	tox -e coverage

docs: ## generate Sphinx HTML documentation, including API docs
	tox -e docs

