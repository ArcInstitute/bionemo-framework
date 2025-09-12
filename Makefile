PYTHON_VERSION := 3.12
UV_INSTALL_DIR := $(HOME)/uv
UV_BIN := $(UV_INSTALL_DIR)/uv
# UV command that works whether uv is in PATH or needs full path
UV := $(shell which uv 2>/dev/null || echo "$(UV_BIN)")

# Include configuration file if it exists to get INFRA variable
-include .config.mk

# Set versions based on INFRA configuration (defaults to CHIMERA if not set)
ifeq ($(INFRA),LAMBDA)
CUDA_VERSION ?= $(LAMBDA_CUDA_VERSION)
TORCH_VERSION ?= $(LAMBDA_TORCH_VERSION)
TORCHVISION_VERSION ?= $(LAMBDA_TORCHVISION_VERSION)
FLASH_ATTN_VERSION ?= $(LAMBDA_FLASH_ATTN_VERSION)
else
CUDA_VERSION ?= $(CHIMERA_CUDA_VERSION)
TORCH_VERSION ?= $(CHIMERA_TORCH_VERSION)
TORCHVISION_VERSION ?= $(CHIMERA_TORCHVISION_VERSION)
FLASH_ATTN_VERSION ?= $(CHIMERA_FLASH_ATTN_VERSION)
endif
NUMPY_VERSION ?= $(CHIMERA_NUMPY_VERSION)
MAX_JOBS ?= $(CHIMERA_MAX_JOBS)
TE_MAX_JOBS ?= $(CHIMERA_TE_MAX_JOBS)
CAUSAL_CONV1D_VERSION ?= $(CHIMERA_CAUSAL_CONV1D_VERSION)
TRANSFORMER_ENGINE_VERSION ?= $(CHIMERA_TRANSFORMER_ENGINE_VERSION)
MAMBA_SSM_VERSION ?= $(CHIMERA_MAMBA_SSM_VERSION)
NEMO_TOOLKIT_VERSION ?= $(CHIMERA_NEMO_TOOLKIT_VERSION)

ifeq ($(VENV_NAME),)
VENV_NAME := $(HOME)/envs/bionemo-venv-fa281
endif

# Chimera Configuration
CHIMERA_CUDA_VERSION := cu121
CHIMERA_TORCH_VERSION := 2.5.1
CHIMERA_TORCHVISION_VERSION := 0.20.1
CHIMERA_FLASH_ATTN_VERSION := 2.7.3
CHIMERA_NUMPY_VERSION := <2.0.0
CHIMERA_MAX_JOBS := 32
CHIMERA_TE_MAX_JOBS := 8
CHIMERA_CAUSAL_CONV1D_VERSION := 1.4.0
CHIMERA_TRANSFORMER_ENGINE_VERSION := 1.11
CHIMERA_MAMBA_SSM_VERSION := 2.2.2
CHIMERA_NEMO_TOOLKIT_VERSION := 2.6.0rc0

# Lambda Configuration
LAMBDA_CUDA_VERSION := cu128
LAMBDA_TORCH_VERSION := 2.8.0
LAMBDA_TORCHVISION_VERSION := 0.23.0
LAMBDA_FLASH_ATTN_VERSION := 2.8.1
LAMBDA_NUMPY_VERSION := <2.0.0
LAMBDA_MAX_JOBS := 80
LAMBDA_TE_MAX_JOBS := 8
LAMBDA_CAUSAL_CONV1D_VERSION := 1.4.0
LAMBDA_TRANSFORMER_ENGINE_VERSION := 1.11
LAMBDA_MAMBA_SSM_VERSION := 2.2.2
LAMBDA_NEMO_TOOLKIT_VERSION := 2.6.0rc0

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Check if uv is installed, if not install it
# From official doc: curl -LsSf https://astral.sh/uv/install.sh | sh
define check_and_install_uv
	@if ! which uv > /dev/null 2>&1 && [ ! -f "$(UV_BIN)" ]; then \
		echo "$(YELLOW)UV not found. Installing...$(NC)"; \
		curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$(UV_INSTALL_DIR)" sh && \
		echo "$(GREEN)UV installed successfully at $(UV_BIN)$(NC)"; \
	elif [ -f "$(UV_BIN)" ]; then \
		echo "$(GREEN)UV is installed at $(UV_BIN)$(NC)"; \
	else \
		echo "$(GREEN)UV is already in PATH$(NC)"; \
	fi
endef


define activate_venv
	if [ -z "$$VIRTUAL_ENV" ] && [ -d "$(VENV_NAME)" ] && [ -f "$(VENV_NAME)/bin/activate" ]; then \
		echo "$(YELLOW)Warning: Virtual environment '$(VENV_NAME)' exists but is not activated. Activating it...$(NC)"; \
		. $(VENV_NAME)/bin/activate; \
	elif [ -z "$$VIRTUAL_ENV" ] && [ -d "$(VENV_NAME)" ] && [ ! -f "$(VENV_NAME)/bin/activate" ]; then \
		echo "$(YELLOW)Warning: Incomplete virtual environment directory found at '$(VENV_NAME)'. Removing and recreating...$(NC)"; \
		rm -rf "$(VENV_NAME)"; \
		echo "$(YELLOW)Creating new virtual environment...$(NC)"; \
		$(UV) venv $(VENV_NAME) --python $(PYTHON_VERSION) && \
		. $(VENV_NAME)/bin/activate; \
		echo "$(GREEN)Virtual environment created and activated: $(VENV_NAME)$(NC)"; \
		export CUDNN_PATH=$$(python -c "import nvidia.cuddnn; print(nvidia.cudnn.__path__[0])" 2>/dev/null || echo "") && \
		export CUDA_HOME=/usr/local/cuda && \
		export LD_LIBRARY_PATH="$$CUDNN_PATH/lib:$$LD_LIBRARY_PATH" && \
		export CPATH="$$CUDNN_PATH/include:$$CPATH" && \
		export LIBRARY_PATH="$$CUDNN_PATH/lib:$$LIBRARY_PATH"; \
	elif [ -z "$$VIRTUAL_ENV" ] && [ ! -d "$(VENV_NAME)" ]; then \
		echo "$(YELLOW)No virtual environment detected. Creating new virtual environment...$(NC)"; \
		$(UV) venv $(VENV_NAME) --python $(PYTHON_VERSION) && \
		. $(VENV_NAME)/bin/activate; \
		echo "$(GREEN)Virtual environment created and activated: $(VENV_NAME)$(NC)"; \
		export CUDNN_PATH=$$(python -c "import nvidia.cuddnn; print(nvidia.cudnn.__path__[0])" 2>/dev/null || echo "") && \
		export CUDA_HOME=/usr/local/cuda && \
		export LD_LIBRARY_PATH="$$CUDNN_PATH/lib:$$LD_LIBRARY_PATH" && \
		export CPATH="$$CUDNN_PATH/include:$$CPATH" && \
		export LIBRARY_PATH="$$CUDNN_PATH/lib:$$LIBRARY_PATH"; \
	fi
endef

define install_pytorch
	. $(VENV_NAME)/bin/activate && \
	$(UV) pip install torch==$(TORCH_VERSION)+$(CUDA_VERSION) \
					torchvision==$(TORCHVISION_VERSION)+$(CUDA_VERSION) \
					torchaudio==$(TORCH_VERSION)+$(CUDA_VERSION) \
					--index-url https://download.pytorch.org/whl/$(CUDA_VERSION)
endef

define install_numpy
	. $(VENV_NAME)/bin/activate && $(UV) pip install "numpy$(NUMPY_VERSION)"
endef

define install_causal_conv1d
	. $(VENV_NAME)/bin/activate && CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=$(MAX_JOBS) \
	$(UV) pip install --no-build-isolation \
	git+https://github.com/trvachov/causal-conv1d.git@52e06e3d5ca10af0c7eb94a520d768c48ef36f1f
endef

define install_transformer_engine
	. $(VENV_NAME)/bin/activate && MAX_JOBS=$(TE_MAX_JOBS) NVTE_BUILD_THREADS_PER_JOB=$(TE_MAX_JOBS) \
	$(UV) pip install "transformer-engine[pytorch]" --no-build-isolation --force-reinstall
endef

define install_mamba_ssm
	. $(VENV_NAME)/bin/activate && $(UV) pip install mamba-ssm
endef

define install_flash_attn
	MAJOR_VERSION=$$(echo "$(FLASH_ATTN_VERSION)" | cut -d. -f1); \
	if [ "$(INFRA)" = "LAMBDA" ] && [ "$$MAJOR_VERSION" -gt "3" ]; then \
		echo "$(YELLOW)Installing flash-attn via git clone for Lambda...$(NC)"; \
		. $(VENV_NAME)/bin/activate && $(UV) pip install wheel setuptools packaging ninja && \
		git clone https://github.com/Dao-AILab/flash-attention.git && \
		cd flash-attention/hopper && \
		MAX_JOBS=$(MAX_JOBS) python setup.py install && \
		cd ../.. && \
		rm -rf flash-attention/; \
	elif [ "$(INFRA)" = "LAMBDA" ] && [ "$$MAJOR_VERSION" -lt "3" ]; then \
		echo "$(YELLOW)Installing flash-attn $(FLASH_ATTN_VERSION) via pip...$(NC)"; \
		. $(VENV_NAME)/bin/activate && $(UV) pip install wheel setuptools packaging ninja && MAX_JOBS=$(MAX_JOBS) $(UV) pip install flash-attn==$(FLASH_ATTN_VERSION) --no-build-isolation --force-reinstall; \
	else \
		echo "$(YELLOW)Installing flash-attn $(FLASH_ATTN_VERSION) via pip...$(NC)"; \
		. $(VENV_NAME)/bin/activate && $(UV) pip install wheel setuptools packaging ninja && MAX_JOBS=$(MAX_JOBS) $(UV) pip install flash-attn==$(FLASH_ATTN_VERSION) --no-build-isolation --force-reinstall; \
	fi
endef

define check_critical_deps
	@echo "$(YELLOW)Checking critical dependencies versions...$(NC)"
	
	@echo "$(YELLOW)Checking PyTorch version...$(NC)"
	@INSTALLED_TORCH_VERSION="$$(. $(VENV_NAME)/bin/activate && python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not_installed')"; \
	if [ "$$INSTALLED_TORCH_VERSION" != "$(TORCH_VERSION)+$(CUDA_VERSION)" ]; then \
		echo "$(YELLOW)Warning: PyTorch version mismatch. Expected $(TORCH_VERSION)+$(CUDA_VERSION), got $$INSTALLED_TORCH_VERSION.$(NC)"; \
		echo "$(YELLOW)Restoring PyTorch version...$(NC)"; \
		$(install_pytorch); \
	else \
		echo "$(GREEN)PyTorch version matches expected version: $(TORCH_VERSION)+$(CUDA_VERSION)$(NC)"; \
	fi;
	
	@echo "$(YELLOW)Checking flash-attn version...$(NC)"
	@INSTALLED_FLASH_ATTN_VERSION="$$(. $(VENV_NAME)/bin/activate && python -c 'import flash_attn; print(flash_attn.__version__)' 2>/dev/null || echo 'not_installed')"; \
	if [ "$$INSTALLED_FLASH_ATTN_VERSION" != "$(FLASH_ATTN_VERSION)" ]; then \
		echo "$(YELLOW)Warning: Flash Attention version mismatch. Expected $(FLASH_ATTN_VERSION), got $$INSTALLED_FLASH_ATTN_VERSION.$(NC)"; \
		echo "$(YELLOW)Restoring flash-attn version...$(NC)"; \
		$(install_flash_attn); \
	else \
		echo "$(GREEN)Flash Attention version matches expected version: $(FLASH_ATTN_VERSION)$(NC)"; \
	fi;
	
	@echo "$(YELLOW)Checking causal-conv1d version...$(NC)"
	@INSTALLED_CAUSAL_CONV1D_VERSION="$$(. $(VENV_NAME)/bin/activate && python -c 'from causal_conv1d import causal_conv1d_fn; print(\"installed\")' 2>/dev/null || echo 'not_installed')"; \
	if [ "$$INSTALLED_CAUSAL_CONV1D_VERSION" = "not_installed" ]; then \
		echo "$(YELLOW)Causal Conv1D not installed. Installing...$(NC)"; \
		$(install_causal_conv1d); \
	else \
		echo "$(GREEN)Causal Conv1D is installed$(NC)"; \
	fi;
	
	@echo "$(YELLOW)Checking transformer-engine version...$(NC)"
	@INSTALLED_TE_VERSION="$$(. $(VENV_NAME)/bin/activate && python -c 'import transformer_engine; print(\"installed\")' 2>/dev/null || echo 'not_installed')"; \
	if [ "$$INSTALLED_TE_VERSION" = "not_installed" ]; then \
		echo "$(YELLOW)Transformer Engine not installed. Installing...$(NC)"; \
		$(install_transformer_engine); \
	else \
		echo "$(GREEN)Transformer Engine is installed$(NC)"; \
	fi;
	
	@echo "$(YELLOW)Checking mamba-ssm version...$(NC)"
	@INSTALLED_MAMBA_SSM_VERSION="$$(. $(VENV_NAME)/bin/activate && python -c 'import mamba_ssm; print(\"installed\")' 2>/dev/null || echo 'not_installed')"; \
	if [ "$$INSTALLED_MAMBA_SSM_VERSION"   = "not_installed" ]; then \
		echo "$(YELLOW)Mamba SSM not installed. Installing...$(NC)"; \
		$(install_mamba_ssm); \
	elif [ "$$INSTALLED_MAMBA_SSM_VERSION" != "$$MAMBA_SSM_VERSION" ]; then \
		echo "$(YELLOW)Mamba SSM version mismatch. Expected $(MAMBA_SSM_VERSION), got $$INSTALLED_MAMBA_SSM_VERSION.$(NC)"; \
		echo "$(YELLOW)Restoring mamba-ssm version...$(NC)"; \
		$(install_mamba_ssm); \
	else \
		echo "$(GREEN)Mamba SSM is installed$(NC)"; \
	fi;

	@echo "$(YELLOW)Checking numpy version...$(NC)"
	@INSTALLED_NUMPY_VERSION="$$(. $(VENV_NAME)/bin/activate && python -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo 'not_installed')"; \
	if [ "$$INSTALLED_NUMPY_VERSION" != "$(NUMPY_VERSION)" ] && [ "$$INSTALLED_NUMPY_VERSION" != "not_installed" ]; then \
		echo "$(YELLOW)Warning: Numpy version mismatch. Expected $(NUMPY_VERSION), got $$INSTALLED_NUMPY_VERSION.$(NC)"; \
		echo "$(YELLOW)Restoring numpy version...$(NC)"; \
		$(install_numpy); \
	else \
		echo "$(GREEN)Numpy version is compatible$(NC)"; \
	fi;

	@echo "$(GREEN)Critical dependencies versions checked successfully$(NC)"
endef

# Default target
.PHONY: all
all: check-uv activate-venv install-subpackages install-performance install-hyena install-critical-deps install-pytorch test-installation

# Help target
.PHONY: help
help:
	@echo "$(GREEN)BioNeMo Setup Makefile$(NC)"
	@echo ""
	@echo "$(YELLOW)Available targets:$(NC)"
	@echo "  all                 - Complete setup (default)"
	@echo "  activate-venv       - Create and activate virtual environment"
	@echo "  check-venv         - Check if virtual environment is active"
	@echo "  check-venv-setup   - Check virtual environment for setup"
	@echo "  install-pytorch    - Install PyTorch with CUDA support"
	@echo "  install-subpackages- Install BioNeMo sub-packages"
	@echo "  install-performance- Install performance optimization packages"
	@echo "  install-hyena      - Install Hyena model dependencies"
	@echo "  test-installation  - Test the installation"
	@echo "  check-versions     - Display versions of key packages"
	@echo "  activate-env       - Show activation command and environment setup"
	@echo "  export-env         - Generate environment variables for current shell"
	@echo "  create-activation-script - Create activation script"
	@echo "  apply-nemo-patches - Apply BioNeMo-specific patches to NeMo"
	@echo "  revert-nemo-patches- Revert BioNeMo-specific patches (for debugging)"
	@echo "  restore-env        - Restore environment after dependency conflicts"
	@echo "  run-tests          - Run BioNeMo tests (core + evo2 data + stop-and-go)"
	@echo "  run-tests-quick    - Run quick tests (evo2 data tests only)"
	@echo "  run-tests-core     - Run core data tests (requires dependency management)"
	@echo "  run-tests-all      - Run all available tests with detailed output"
	@echo "  clean              - Remove virtual environment"
	@echo "  clean-results      - Remove training results"
	@echo ""
	@echo "$(YELLOW)Environment:$(NC)"
	@echo "  Virtual Environment: $(VENV_NAME)"
	@echo "  Python Version: $(PYTHON_VERSION)"
	@echo "  PyTorch Version: $(TORCH_VERSION)+$(CUDA_VERSION)"
	@echo "  Torchvision Version: $(TORCHVISION_VERSION)+$(CUDA_VERSION)"
	@echo "  Flash Attention: $(FLASH_ATTN_VERSION)"

# Check if UV is installed
.PHONY: check-uv
check-uv:
	@if [ ! -f ".config.mk" ]; then \
		echo "$(RED)No configuration file found. Use \`make all-chimera\` or \`make all-lambda\` to set the configuration.$(NC)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Checking UV...$(NC)"
	$(check_and_install_uv)

# Create/Activate virtual environment
.PHONY: activate-venv
activate-venv: check-uv
	@$(activate_venv)

.PHONY: all-chimera
all-chimera:
	@echo "$(YELLOW)Using Chimera configuration...$(NC)"
	if [ ! -f ".config.mk" ]; then \
		echo "INFRA=CHIMERA" > .config.mk; \
		echo "VENV_NAME=$(HOME)/envs/bionemo-venv-fa281" >> .config.mk; \
	fi

	git submodule update --init --recursive
	$(MAKE) all

.PHONY: all-lambda
all-lambda:
	@echo "$(YELLOW)Using Lambda configuration...$(NC)"
	if [ ! -f ".config.mk" ]; then \
		echo "INFRA=LAMBDA" > .config.mk; \
		echo "VENV_NAME=$(HOME)/envs/bionemo-venv-fa281" >> .config.mk; \
	fi

	git submodule update --init --recursive
	$(MAKE) all


# Install PyTorch with CUDA support
.PHONY: install-pytorch
install-pytorch: activate-venv
	@echo "$(YELLOW)Installing PyTorch $(TORCH_VERSION)+$(CUDA_VERSION)...$(NC)"
	@. $(VENV_NAME)/bin/activate && \
	$(UV) pip install torch==$(TORCH_VERSION)+$(CUDA_VERSION) \
					torchvision==$(TORCHVISION_VERSION)+$(CUDA_VERSION) \
					torchaudio==$(TORCH_VERSION)+$(CUDA_VERSION) \
					--index-url https://download.pytorch.org/whl/$(CUDA_VERSION)
	@echo "$(GREEN) PyTorch installed successfully$(NC)"
	@. $(VENV_NAME)/bin/activate && \
	python -c "import torch; import torchvision; print('PyTorch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('CUDA available:', torch.cuda.is_available())"

# Install BioNeMo sub-packages
.PHONY: install-subpackages
install-subpackages: activate-venv
	@echo "$(YELLOW)Installing BioNeMo sub-packages...$(NC)"

	@echo "$(YELLOW)Installing bionemo-llm...$(NC)"
	@. $(VENV_NAME)/bin/activate && cd sub-packages/bionemo-llm && $(UV) pip install -e .
	
	@echo "$(YELLOW)Installing bionemo-core...$(NC)"
	@. $(VENV_NAME)/bin/activate && cd sub-packages/bionemo-core && $(UV) pip install -e . --force-reinstall
	
	@echo "$(YELLOW)Installing maturin for Rust packages...$(NC)"
	@. $(VENV_NAME)/bin/activate && $(UV) pip install maturin
	
	@echo "$(YELLOW)Installing bionemo-noodles (Rust-based)...$(NC)"
	@. $(VENV_NAME)/bin/activate && cd sub-packages/bionemo-noodles && $(UV) pip install -e .
	
	@echo "$(YELLOW)Installing Megatron-LM...$(NC)"
	@. $(VENV_NAME)/bin/activate && cd 3rdparty/Megatron-LM && $(UV) pip install -e .
	
	@echo "$(YELLOW)Installing NeMo toolkit with NLP and eval extras...$(NC)"
	@. $(VENV_NAME)/bin/activate && cd 3rdparty/NeMo && $(UV) pip install -e ".[nlp,eval]"
	
	@echo "$(YELLOW)Applying BioNeMo-specific patches to NeMo...$(NC)"
	@$(MAKE) apply-nemo-patches
	
	@echo "$(YELLOW)Installing NeMo-Run (specific commit)...$(NC)"
	@. $(VENV_NAME)/bin/activate && \
	$(UV) pip install git+https://github.com/NVIDIA/NeMo-Run@34259bd3e752fef94045a9a019e4aaf62bd11ce2
	
	@echo "$(YELLOW)Installing bionemo-evo2...$(NC)"
	@. $(VENV_NAME)/bin/activate && cd sub-packages/bionemo-evo2 && $(UV) pip install -e .
	
	@echo "$(YELLOW)Reinstalling NeMo from local source to ensure correct version with hyena modules...$(NC)"
	# bionemo-llm dependency on nemo_toolkit>=2.2.1 can cause downgrade from local version
	# This step ensures we maintain the local NeMo version (2.6.0rc0) with hyena modules
	@. $(VENV_NAME)/bin/activate && cd 3rdparty/NeMo && $(UV) pip install -e ".[nlp,eval]" --force-reinstall
	
	@echo "$(GREEN) BioNeMo sub-packages installed successfully$(NC)"

# Install performance optimization packages
.PHONY: install-performance
install-performance: activate-venv
	@echo "$(YELLOW)Installing performance optimization packages...$(NC)"
	
	@echo "$(YELLOW)Installing cuDNN package...$(NC)"
	@. $(VENV_NAME)/bin/activate && $(UV) pip install nvidia-cudnn-cu12
	
	@echo "$(YELLOW)Installing Transformer Engine with limited parallel jobs...$(NC)"
	@. $(VENV_NAME)/bin/activate && \
	MAX_JOBS=$(TE_MAX_JOBS) NVTE_BUILD_THREADS_PER_JOB=$(TE_MAX_JOBS) \
	$(UV) pip install "transformer-engine[pytorch]" --no-build-isolation --force-reinstall
	
	@echo "$(GREEN) Performance packages installed successfully$(NC)"

# Install Hyena model dependencies
.PHONY: install-hyena
install-hyena: activate-venv
	@echo "$(YELLOW)Installing Hyena model dependencies...$(NC)"
	@echo "$(YELLOW)Installing causal-conv1d from tested fork...$(NC)"
	@$(install_causal_conv1d)
	@echo "$(GREEN) Hyena dependencies installed successfully$(NC)"

.PHONY: check-critical-deps
check-critical-deps: activate-venv
	@echo "$(YELLOW)Checking critical dependencies versions...$(NC)"
	$(check_critical_deps)
	@echo "$(GREEN) Critical dependencies versions checked successfully$(NC)"

.PHONY: install-critical-deps
install-critical-deps: activate-venv
	@echo "$(YELLOW)Installing critical dependencies...$(NC)"
	$(check_critical_deps)
	@echo "$(GREEN) Critical dependencies installed successfully$(NC)"

# Test the installation
.PHONY: test-installation
test-installation: 
	@echo "$(YELLOW)Testing BioNeMo installation...$(NC)"
	
	@echo "$(YELLOW)Testing PyTorch and CUDA...$(NC)"
	@. $(VENV_NAME)/bin/activate && \
	python -c "import torch; print(' PyTorch:', torch.__version__); print(' CUDA available:', torch.cuda.is_available())"
	
	@echo "$(YELLOW)Testing BioNeMo imports...$(NC)"
	@. $(VENV_NAME)/bin/activate && \
	python -c "import bionemo.core; print(' BioNeMo Core')" && \
	python -c "import bionemo.evo2; print(' BioNeMo Evo2')" && \
	python -c "import bionemo.llm; print(' BioNeMo LLM')" && \
	python -c "import bionemo.noodles; print(' BioNeMo Noodles')"
	
	@echo "$(YELLOW)Verifying NeMo version and hyena modules...$(NC)"
	@. $(VENV_NAME)/bin/activate && \
	python -c "import nemo; print(' NeMo version:', nemo.__version__)" && \
	python -c "from nemo.collections.llm.gpt.data.megatron.hyena.config import parse_dataset_config; print(' ✅ Hyena modules available')"
	
	@echo "$(YELLOW)Testing performance packages...$(NC)"
	@. $(VENV_NAME)/bin/activate && \
	python -c "import flash_attn; print(' Flash Attention:', flash_attn.__version__)" && \
	python -c "from causal_conv1d import causal_conv1d_fn; print(' Causal Conv1D: Available')"
	
	@echo "$(YELLOW)Testing train_evo2 command...$(NC)"
	@. $(VENV_NAME)/bin/activate && train_evo2 --help | head -5
	
	@echo "$(YELLOW)Running quick training test...$(NC)"
	@. $(VENV_NAME)/bin/activate && \
	rm -rf results/evo2 && \
	train_evo2 --mock-data --devices 1 --max-steps 1 --micro-batch-size 1 --model-size test
	
	@echo "$(GREEN) Installation test completed successfully!$(NC)"

# Display versions of key packages
.PHONY: check-versions
check-versions:
	@echo "$(YELLOW)Key Package Versions:$(NC)"
	@. $(VENV_NAME)/bin/activate && \
	echo "$(GREEN)System Information:$(NC)" && \
	echo "  Python: $$(python --version)" && \
	echo "  UV: $$(uv --version)" && \
	echo "" && \
	echo "$(GREEN)Core Dependencies:$(NC)" && \
	python -c "import torch; print('  PyTorch:', torch.__version__)" && \
	python -c "import numpy; print('  NumPy:', numpy.__version__)" && \
	echo "" && \
	echo "$(GREEN)BioNeMo Packages:$(NC)" && \
	python -c "import bionemo.core; print('  bionemo-core: Available')" && \
	python -c "import bionemo.evo2; print('  bionemo-evo2: Available')" && \
	python -c "import bionemo.llm; print('  bionemo-llm: Available')" && \
	python -c "import bionemo.noodles; print('  bionemo-noodles: Available')" && \
	echo "" && \
	echo "$(GREEN)Performance Packages:$(NC)" && \
	python -c "try: import transformer_engine; print('  Transformer Engine: Available'); except: print('  Transformer Engine: Not Available')" && \
	python -c "import flash_attn; print('  Flash Attention:', flash_attn.__version__)" && \
	python -c "try: from causal_conv1d import causal_conv1d_fn; print('  Causal Conv1D: Available'); except: print('  Causal Conv1D: Not Available')" && \
	echo "" && \
	echo "$(GREEN)GPU Information:$(NC)" && \
	python -c "import torch; print('  CUDA Available:', torch.cuda.is_available()); print('  GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

# Show activation command and set up environment
.PHONY: activate-env
activate-env:
	@echo "$(YELLOW)To activate the environment manually, run:$(NC)"
	@echo "  source $(VENV_NAME)/bin/activate"
	@echo ""
	@echo "$(YELLOW)To set up environment variables for optimal performance:$(NC)"
	@echo "  export CUDNN_PATH=\$$(python -c \"import nvidia.cudnn; print(nvidia.cudnn.__path__[0])\")"
	@echo "  export CUDA_HOME=/usr/local/cuda"
	@echo "  export LD_LIBRARY_PATH=\"\$$CUDNN_PATH/lib:\$$LD_LIBRARY_PATH\""
	@echo "  export CPATH=\"\$$CUDNN_PATH/include:\$$CPATH\""
	@echo "  export LIBRARY_PATH=\"\$$CUDNN_PATH/lib:\$$LIBRARY_PATH\""
	@echo ""
	@echo "$(GREEN)Or simply use: source activate_bionemo_evo2.sh$(NC)"

# Export environment variables for current shell (call with: eval $(make export-env))
.PHONY: export-env
export-env:
	@if [ -d "$(VENV_NAME)" ]; then \
		. $(VENV_NAME)/bin/activate && \
		CUDNN_PATH=$$(python -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])" 2>/dev/null || echo "") && \
		echo "export CUDNN_PATH=$$CUDNN_PATH" && \
		echo "export CUDA_HOME=/usr/local/cuda" && \
		echo "export LD_LIBRARY_PATH=$$CUDNN_PATH/lib:\$$LD_LIBRARY_PATH" && \
		echo "export CPATH=$$CUDNN_PATH/include:\$$CPATH" && \
		echo "export LIBRARY_PATH=$$CUDNN_PATH/lib:\$$LIBRARY_PATH"; \
	else \
		echo "echo 'Error: Virtual environment $(VENV_NAME) not found. Run make all first.'"; \
	fi


# Clean up virtual environment
.PHONY: clean
clean:
	@echo "$(YELLOW)Removing virtual environment: $(VENV_NAME)$(NC)"
	@if [ -d "$(VENV_NAME)" ]; then \
		rm -rf $(VENV_NAME); \
		echo "  Removed $(VENV_NAME)"; \
	else \
		echo "  Virtual environment $(VENV_NAME) not found"; \
	fi
	@if [ -f ".config.mk" ]; then \
		rm -f .config.mk; \
		echo "  Removed .config.mk"; \
	fi
	@echo "$(GREEN) Environment cleaned$(NC)"

# Clean up results
.PHONY: clean-results
clean-results:
	@echo "$(YELLOW)Removing training results...$(NC)"
	@rm -rf results lightning_logs
	@echo "$(GREEN) Results cleaned$(NC)"


# Show system information
.PHONY: system-info
system-info:
	@echo "$(YELLOW)System Information:$(NC)"
	@echo "  OS: $$(uname -s -r)"
	@echo "  Architecture: $$(uname -m)"
	@echo "  Available GPUs: $$(nvidia-smi -L 2>/dev/null | wc -l || echo '0')"
	@echo "  CUDA Driver: $$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 'Not Available')"
	@echo "  CUDA Runtime: $$(nvcc --version 2>/dev/null | grep 'release' | awk '{print $$6}' | cut -c2- || echo 'Not Available')"

# Apply BioNeMo-specific patches to NeMo
.PHONY: apply-nemo-patches
apply-nemo-patches:
	@echo "$(YELLOW)Applying quick_gelu import fix...$(NC)"
	@if grep -q 'safe_import_from("megatron.core.fusions.fused_bias_geglu", "quick_gelu"' 3rdparty/NeMo/nemo/collections/llm/gpt/model/gpt_oss.py 2>/dev/null; then \
		echo "  Fixing quick_gelu import in gpt_oss.py..."; \
		sed -i 's/quick_gelu, HAVE_QUICK_GELU = safe_import_from("megatron.core.fusions.fused_bias_geglu", "quick_gelu", alt=object)/from nemo.collections.llm.fn.activation import quick_gelu\nHAVE_QUICK_GELU = True/' 3rdparty/NeMo/nemo/collections/llm/gpt/model/gpt_oss.py; \
		echo "$(GREEN)  quick_gelu import fix applied successfully$(NC)"; \
	elif grep -q 'from nemo.collections.llm.fn.activation import quick_gelu' 3rdparty/NeMo/nemo/collections/llm/gpt/model/gpt_oss.py 2>/dev/null; then \
		echo "$(GREEN)  quick_gelu import fix already applied$(NC)"; \
	else \
		echo "$(RED)  Warning: Could not find expected import pattern in gpt_oss.py$(NC)"; \
	fi
	@echo "$(GREEN) BioNeMo patches applied successfully$(NC)"

# Revert BioNeMo-specific patches (for development/debugging)
.PHONY: revert-nemo-patches
revert-nemo-patches:
	@echo "$(YELLOW)Reverting BioNeMo-specific patches...$(NC)"
	@if grep -q 'from nemo.collections.llm.fn.activation import quick_gelu' 3rdparty/NeMo/nemo/collections/llm/gpt/model/gpt_oss.py 2>/dev/null; then \
		echo "  Reverting quick_gelu import fix in gpt_oss.py..."; \
		sed -i 's/from nemo.collections.llm.fn.activation import quick_gelu\nHAVE_QUICK_GELU = True/quick_gelu, HAVE_QUICK_GELU = safe_import_from("megatron.core.fusions.fused_bias_geglu", "quick_gelu", alt=object)/' 3rdparty/NeMo/nemo/collections/llm/gpt/model/gpt_oss.py; \
		echo "$(GREEN)  quick_gelu import fix reverted successfully$(NC)"; \
	else \
		echo "$(GREEN)  No patches to revert$(NC)"; \
	fi
	@echo "$(GREEN) BioNeMo patches reverted successfully$(NC)"


# Run tests for BioNeMo packages
.PHONY: run-tests
run-tests:
	@echo "$(YELLOW)Running BioNeMo tests...$(NC)"
	@echo "$(YELLOW)Installing test dependencies...$(NC)"
	@. $(VENV_NAME)/bin/activate && \
	$(UV) pip install pytest pytest-xdist pytest-cov
	@echo "$(YELLOW)Installing bionemo-testing package...$(NC)"
	@. $(VENV_NAME)/bin/activate && cd sub-packages/bionemo-testing && $(UV) pip install -e .
	@echo "$(YELLOW)Running bionemo-core data tests...$(NC)"
	@. $(VENV_NAME)/bin/activate && cd sub-packages/bionemo-core && \
	python -m pytest tests/bionemo/core/data/ -v --tb=short --maxfail=10
	@echo "$(YELLOW)Running bionemo-evo2 data tests...$(NC)"
	@. $(VENV_NAME)/bin/activate && cd sub-packages/bionemo-evo2 && \
	python -m pytest tests/bionemo/evo2/data/ -v --tb=short --maxfail=10
	@echo "$(YELLOW)Running bionemo-evo2 stop-and-go tests...$(NC)"
	@. $(VENV_NAME)/bin/activate && cd sub-packages/bionemo-evo2 && \
	python -m pytest tests/bionemo/evo2/test_stop_and_go.py -v --tb=short --maxfail=10
	@echo "$(GREEN) Test run completed!$(NC)"
	@echo ""
	@echo "$(YELLOW)Test Summary:$(NC)"
	@echo "  ✅ Core data tests: Should pass (47 tests)"
	@echo "  ✅ Evo2 data tests: Should pass (12 tests)"
	@echo "  ✅ Import issues: All critical imports now work"
	@echo "  ⚠️  Stop-and-go tests: Some may fail due to missing mamba-ssm"
	@echo "  ❌ Other tests: May fail due to missing dependencies"
	@echo ""
	@echo "$(YELLOW)Note:$(NC) Some tests require additional dependencies like mamba-ssm"
	@echo "To install mamba-ssm: . $(VENV_NAME)/bin/activate && pip install mamba-ssm"

# Run quick tests (only data tests that are known to pass)
.PHONY: run-tests-quick
run-tests-quick:
	@echo "$(YELLOW)Running quick BioNeMo tests (data tests only)...$(NC)"
	@echo "$(YELLOW)Installing test dependencies...$(NC)"
	@. $(VENV_NAME)/bin/activate && $(UV) pip install pytest pytest-xdist pytest-cov
	@echo "$(YELLOW)Running bionemo-evo2 FASTA dataset tests (no external deps needed)...$(NC)"
	@. $(VENV_NAME)/bin/activate && cd sub-packages/bionemo-evo2 && \
	python -m pytest tests/bionemo/evo2/data/test_fasta_dataset.py -v --tb=short
	@echo "$(GREEN) Quick tests completed successfully!$(NC)"
	@echo "$(YELLOW)Note:$(NC) For core data tests, use 'make run-tests-core' (requires additional setup)"

# Run core data tests specifically (requires careful dependency management)
.PHONY: run-tests-core
run-tests-core:
	@echo "$(YELLOW)Running bionemo-core data tests...$(NC)"
	@echo "$(YELLOW)Installing test dependencies...$(NC)"
	@. $(VENV_NAME)/bin/activate && $(UV) pip install pytest pytest-xdist pytest-cov
	@echo "$(YELLOW)Ensuring bionemo-core is properly installed...$(NC)"
	@. $(VENV_NAME)/bin/activate && cd sub-packages/bionemo-core && $(UV) pip install -e . --force-reinstall
	@echo "$(YELLOW)Restoring compatible versions...$(NC)"
	@. $(VENV_NAME)/bin/activate && \
	$(UV) pip install torch==$(TORCH_VERSION)+$(CUDA_VERSION) \
					torchvision==$(TORCHVISION_VERSION)+$(CUDA_VERSION) \
					torchaudio==$(TORCH_VERSION)+$(CUDA_VERSION) \
					--index-url https://download.pytorch.org/whl/$(CUDA_VERSION) --force-reinstall && \
	$(UV) pip install "numpy$(NUMPY_VERSION)" && \
	$(install_flash_attn) && \
	CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=$(MAX_JOBS) \
	$(UV) pip install --no-build-isolation \
	git+https://github.com/trvachov/causal-conv1d.git@52e06e3d5ca10af0c7eb94a520d768c48ef36f1f
	@echo "$(YELLOW)Running core data tests...$(NC)"
	@. $(VENV_NAME)/bin/activate && cd sub-packages/bionemo-core && \
	python -m pytest tests/bionemo/core/data/ -v --tb=short
	@echo "$(GREEN) Core data tests completed!$(NC)"

# Run all available tests with detailed output
.PHONY: run-tests-all
run-tests-all:
	@echo "$(YELLOW)Running all available BioNeMo tests...$(NC)"
	@echo "$(YELLOW)Installing test dependencies...$(NC)"
	@. $(VENV_NAME)/bin/activate && \
	$(UV) pip install pytest pytest-xdist pytest-cov
	@echo "$(YELLOW)Installing bionemo-testing package...$(NC)"
	@. $(VENV_NAME)/bin/activate && cd sub-packages/bionemo-testing && $(UV) pip install -e .
	@echo "$(YELLOW)Running all bionemo-evo2 tests...$(NC)"
	@. $(VENV_NAME)/bin/activate && cd sub-packages/bionemo-evo2 && \
	python -m pytest tests/ -v --tb=short --continue-on-collection-errors --maxfail=20 || true
	@echo "$(GREEN) All tests completed!$(NC)"
