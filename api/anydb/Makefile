# Use bash as the shell
SHELL := /bin/bash

# Variables
CONDA_PATH = /opt/anaconda3/etc/profile.d/conda.sh

# Conda test environment name
ENV_NAME = py10
BASE_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
WORKSPACE = $(BASE_DIR)

setup_environment:
	source $(CONDA_PATH) && conda activate $(ENV_NAME)
	export PYTHONPATH=$(WORKSPACE):$$PYTHONPATH

test:
	pytest --capture=sys -v .
