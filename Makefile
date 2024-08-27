SHELL := /bin/bash
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate
current_dir = $(shell pwd)

PROJECT = biogen
n ?= auto

# The Python version to use for the environment.
PYTHON_VERSION ?= 3.10
# Either "conda" or "mamba".
PACKAGE_MANAGER ?= conda
# Runs a command in the project environment.
RUN_IN_ENV = $(PACKAGE_MANAGER) run --name $(PROJECT)

# By default, enable CUDA except on macOS.
# User can override by setting the CUDA variable.
OS = $(shell uname)
ARCH = $(shell uname -m)
ifeq ($(OS), Darwin)  # macOS
	CUDA ?= 0
else  # not macOS (e.g., Windows, Linux)
	CUDA ?= 1
endif

ifeq ($(CUDA), 1)
    CUDA_VERSION ?= 11.7
    CUDA_PACKAGES = cudatoolkit=$(CUDA_VERSION)
	DOCKER_GPUS = --gpus all
endif

CONDA_PACKAGES = conda-lock mamba $(CUDA_PACKAGES)

# This command will automatically install python 3.10 and base packages like pip, pip3
.POSIX:
env:
	@echo " --- --- --- --- --- --- --- --- --- --- "
	@echo "Creating conda env named $(PROJECT) ... "
	@echo " --- --- --- --- --- --- --- --- --- --- "
	$(PACKAGE_MANAGER) create --name $(PROJECT) python=$(PYTHON_VERSION)

# This command will install all necessary packages in the environment
.POSIX:
install:
	@echo " --- --- --- --- --- --- --- --- --- --- "
	@echo "Installing packages in $(PROJECT) ... "
	@echo " --- --- --- --- --- --- --- --- --- --- "
	$(RUN_IN_ENV) pip install -r requirements.txt
	$(RUN_IN_ENV) pip install -e .


.POSIX:
style:
	black src scripts
	isort src scripts --profile black

.POSIX:
check:
	!(grep -R /tmp tests)
	flakehell lint src/${PROJECT} scripts
	pylint src/${PROJECT}
	black --check src/${PROJECT} notebooks scripts

.PHONY: docs
docs:
	rm -rf ./docs/source/notebooks
	cp -r ./notebooks ./docs/source/notebooks
	docker run --rm -v ${current_dir}:/${PROJECT} --network host -w /${PROJECT}/docs ${DOCKER_ORG}/${PROJECT}:${VERSION} make html
	python3 -m http.server --directory  ./docs/build/html/

.PHONY: test
test:
	find . -name "*.pyc" -delete
	pytest tests -n $n -s -o log_cli=true -o log_cli_level=info

.PHONY: test-codecov
test-codecov:
	find . -name "*.pyc" -delete
	pytest tests -n $n -s -o log_cli=true -o log_cli_level=info --cov=./src/ --cov-report=xml --cov-config=pyproject.toml
