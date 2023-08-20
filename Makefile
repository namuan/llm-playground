export PROJECTNAME=$(shell basename "$(PWD)")

.SILENT: ;               # no need for @

setup: ## Setup Virtual Env
	python3.11 -m venv venv
	./venv/bin/pip3 install -r requirements.txt
	./venv/bin/python3 -m pip install --upgrade pip

deps: ## Install dependencies
	./venv/bin/pip3 install --upgrade -r requirements.txt
	./venv/bin/python3 -m pip install --upgrade pip

clean: ## Clean package
	find . -type d -name '__pycache__' | xargs rm -rf
	rm -rf build dist

pre-commit: ## Manually run all precommit hooks
	./venv/bin/pre-commit install
	./venv/bin/pre-commit run --all-files

pre-commit-tool: ## Manually run a single pre-commit hook
	./venv/bin/pre-commit run $(TOOL) --all-files

build: clean pre-commit ## Build package
	echo "✅ Done"

bpython: ## Runs bpython
	./venv/bin/bpython

.PHONY: help
.DEFAULT_GOAL := help

help: Makefile
	echo
	echo " Choose a command run in "$(PROJECTNAME)":"
	echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
	echo
