# Global variables
PYTHON_INTERPRETER = python3

export PATH := $(shell pwd)/:$(PATH)

.PHONY: clean-build
clean-build: ## Remove build artifacts
	@echo "+ $@"
	@rm -fr build/
	@rm -fr dist/
	@rm -fr *.egg-info

.PHONY: clean-pyc
clean-pyc: ## Remove Python file artifacts
	@echo "+ $@"
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@find . -type f -name '*.py[co]' -exec rm -f {} +
	@find . -name '*~' -exec rm -f {} +

.PHONY: docs
docs: ## Rebuild docs automatically and display locally.
	cd docs && mkdocs serve

.PHONY: servedocs
servedocs: ## Rebuild docs automatically and push to github.
	mkdocs build --config-file docs/mkdocs.yml
	cp -rf docs/site/* .
	rm -rf docs/site
	git add --all
	git commit -m "Updates to Website"
	git push origin master
	@echo "Website updated! Check it out: https://gmihaila.github.io "

.PHONY: setup_marp
setup_marp: ## Download Marp locally and setup.
	bash setup_marp.sh
	@echo "Use ./marp --help"

.PHONY: marp
marp: ## Serve Marp locally.
	./marp -s ./docs/markdown/activities

.PHONY: servemarp
servemarp: ## Convert markdowns to slides using Marp.
	./marp --input-dir ./docs/markdown/activities/

.PHONY: help
help: ## Display make help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'