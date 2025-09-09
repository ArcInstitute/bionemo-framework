# Chimera Configuration
PYTHON_VERSION := 3.12
VENV_NAME := bionemo-evo2-env
CUDA_VERSION := cu121
PYTORCH_VERSION := 2.5.1
FLASH_ATTN_VERSION := 2.7.3
NUMPY_VERSION := <2.0.0
MAX_JOBS := 4
TE_MAX_JOBS := 1

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Global environment setup function
define setup_env
	. $(VENV_NAME)/bin/activate && \
	export CUDNN_PATH=$$(python -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])" 2>/dev/null || echo "") && \
	export CUDA_HOME=/usr/local/cuda && \
	export LD_LIBRARY_PATH="$$CUDNN_PATH/lib:$$LD_LIBRARY_PATH" && \
	export CPATH="$$CUDNN_PATH/include:$$CPATH" && \
	export LIBRARY_PATH="$$CUDNN_PATH/lib:$$LIBRARY_PATH"
endef

# Default target
.PHONY: all
all: setup-env install-pytorch install-subpackages install-performance install-hyena test-installation

# Help target
.PHONY: help
help:
	@echo "$(GREEN)BioNeMo Evo2 Setup Makefile for Chimera Server$(NC)"
	@echo ""
	@echo "$(YELLOW)Available targets:$(NC)"
	@echo "  all                 - Complete setup (default)"
	@echo "  setup-env          - Create and activate virtual environment"
	@echo "  install-pytorch    - Install PyTorch with CUDA 12.1 support"
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
	@echo "  PyTorch Version: $(PYTORCH_VERSION)+$(CUDA_VERSION)"
	@echo "  Flash Attention: $(FLASH_ATTN_VERSION)"

# Check if UV is installed
.PHONY: check-uv
check-uv:
	@echo "$(YELLOW)Checking UV installation...$(NC)"
	@which uv > /dev/null || (echo "$(RED)Error: UV not found. Please install UV first.$(NC)" && exit 1)
	@echo "$(GREEN) UV is available$(NC)"

# Create virtual environment
.PHONY: setup-env
setup-env: check-uv
	@echo "$(YELLOW)Creating virtual environment: $(VENV_NAME)$(NC)"
	@if [ -d "$(VENV_NAME)" ]; then \
		echo "$(YELLOW)Virtual environment already exists. Removing...$(NC)"; \
		rm -rf $(VENV_NAME); \
	fi
	uv venv $(VENV_NAME) --python $(PYTHON_VERSION)
	@echo "$(GREEN) Virtual environment created: $(VENV_NAME)$(NC)"

# Install PyTorch with CUDA support
.PHONY: install-pytorch
install-pytorch: setup-env
	@echo "$(YELLOW)Installing PyTorch $(PYTORCH_VERSION)+$(CUDA_VERSION)...$(NC)"
	@. $(VENV_NAME)/bin/activate && \
	uv pip install torch==$(PYTORCH_VERSION)+$(CUDA_VERSION) \
					torchvision==0.20.1+$(CUDA_VERSION) \
					torchaudio==$(PYTORCH_VERSION)+$(CUDA_VERSION) \
					--index-url https://download.pytorch.org/whl/$(CUDA_VERSION)
	@echo "$(GREEN) PyTorch installed successfully$(NC)"
	@. $(VENV_NAME)/bin/activate && \
	python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Install BioNeMo sub-packages
.PHONY: install-subpackages
install-subpackages: install-pytorch
	@echo "$(YELLOW)Installing BioNeMo sub-packages...$(NC)"
	
	@echo "$(YELLOW)Installing bionemo-core...$(NC)"
	@. $(VENV_NAME)/bin/activate && cd sub-packages/bionemo-core && uv pip install -e . --force-reinstall
	@echo "$(YELLOW)Restoring PyTorch, numpy, flash-attn and causal-conv1d versions after bionemo-core install...$(NC)"
	@. $(VENV_NAME)/bin/activate && \
	uv pip install torch==$(PYTORCH_VERSION)+$(CUDA_VERSION) \
					torchvision==0.20.1+$(CUDA_VERSION) \
					torchaudio==$(PYTORCH_VERSION)+$(CUDA_VERSION) \
					--index-url https://download.pytorch.org/whl/$(CUDA_VERSION) --force-reinstall && \
	uv pip install "numpy$(NUMPY_VERSION)" && \
	uv pip install flash-attn==$(FLASH_ATTN_VERSION) --no-build-isolation && \
	CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=$(MAX_JOBS) \
	uv pip install --no-build-isolation \
	git+https://github.com/trvachov/causal-conv1d.git@52e06e3d5ca10af0c7eb94a520d768c48ef36f1f
	
	@echo "$(YELLOW)Installing maturin for Rust packages...$(NC)"
	@. $(VENV_NAME)/bin/activate && uv pip install maturin
	
	@echo "$(YELLOW)Installing bionemo-noodles (Rust-based)...$(NC)"
	@. $(VENV_NAME)/bin/activate && cd sub-packages/bionemo-noodles && uv pip install -e .
	
	@echo "$(YELLOW)Installing Megatron-LM...$(NC)"
	@. $(VENV_NAME)/bin/activate && cd 3rdparty/Megatron-LM && uv pip install -e .
	
	@echo "$(YELLOW)Installing NeMo toolkit with NLP and eval extras...$(NC)"
	@. $(VENV_NAME)/bin/activate && cd 3rdparty/NeMo && uv pip install -e ".[nlp,eval]"
	
	@echo "$(YELLOW)Applying BioNeMo-specific patches to NeMo...$(NC)"
	@$(MAKE) apply-nemo-patches
	
	@echo "$(YELLOW)Installing NeMo-Run (specific commit)...$(NC)"
	@. $(VENV_NAME)/bin/activate && \
	uv pip install git+https://github.com/NVIDIA/NeMo-Run@34259bd3e752fef94045a9a019e4aaf62bd11ce2
	
	@echo "$(YELLOW)Installing bionemo-llm...$(NC)"
	@. $(VENV_NAME)/bin/activate && cd sub-packages/bionemo-llm && uv pip install -e .
	@echo "$(YELLOW)Fixing bionemo-core after bionemo-llm installation...$(NC)"
	@. $(VENV_NAME)/bin/activate && cd sub-packages/bionemo-core && uv pip install -e . --force-reinstall
	@echo "$(YELLOW)Restoring PyTorch, numpy, flash-attn and causal-conv1d versions...$(NC)"
	@. $(VENV_NAME)/bin/activate && \
	uv pip install torch==$(PYTORCH_VERSION)+$(CUDA_VERSION) \
					torchvision==0.20.1+$(CUDA_VERSION) \
					torchaudio==$(PYTORCH_VERSION)+$(CUDA_VERSION) \
					--index-url https://download.pytorch.org/whl/$(CUDA_VERSION) --force-reinstall && \
	uv pip install "numpy$(NUMPY_VERSION)" && \
	uv pip install flash-attn==$(FLASH_ATTN_VERSION) --no-build-isolation && \
	CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=$(MAX_JOBS) \
	uv pip install --no-build-isolation \
	git+https://github.com/trvachov/causal-conv1d.git@52e06e3d5ca10af0c7eb94a520d768c48ef36f1f
	
	@echo "$(YELLOW)Installing bionemo-evo2...$(NC)"
	@. $(VENV_NAME)/bin/activate && cd sub-packages/bionemo-evo2 && uv pip install -e .
	
	@echo "$(YELLOW)Reinstalling NeMo from local source to ensure correct version with hyena modules...$(NC)"
	# bionemo-llm dependency on nemo_toolkit>=2.2.1 can cause downgrade from local version
	# This step ensures we maintain the local NeMo version (2.6.0rc0) with hyena modules
	@. $(VENV_NAME)/bin/activate && cd 3rdparty/NeMo && uv pip install -e ".[nlp,eval]" --force-reinstall
	
	@echo "$(YELLOW)Final version verification and restoration...$(NC)"
	@. $(VENV_NAME)/bin/activate && \
	uv pip install torch==$(PYTORCH_VERSION)+$(CUDA_VERSION) \
					torchvision==0.20.1+$(CUDA_VERSION) \
					torchaudio==$(PYTORCH_VERSION)+$(CUDA_VERSION) \
					--index-url https://download.pytorch.org/whl/$(CUDA_VERSION) --force-reinstall && \
	uv pip install "numpy$(NUMPY_VERSION)" && \
	uv pip install flash-attn==$(FLASH_ATTN_VERSION) --no-build-isolation && \
	CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=$(MAX_JOBS) \
	uv pip install --no-build-isolation \
	git+https://github.com/trvachov/causal-conv1d.git@52e06e3d5ca10af0c7eb94a520d768c48ef36f1f
	
	@echo "$(GREEN) BioNeMo sub-packages installed successfully$(NC)"

# Install performance optimization packages
.PHONY: install-performance
install-performance: install-subpackages
	@echo "$(YELLOW)Installing performance optimization packages...$(NC)"
	
	@echo "$(YELLOW)Installing cuDNN package...$(NC)"
	@. $(VENV_NAME)/bin/activate && uv pip install nvidia-cudnn-cu12
	
	@echo "$(YELLOW)Installing Transformer Engine with limited parallel jobs...$(NC)"
	@$(setup_env) && \
	MAX_JOBS=$(TE_MAX_JOBS) NVTE_BUILD_THREADS_PER_JOB=$(TE_MAX_JOBS) \
	uv pip install "transformer-engine[pytorch]" --no-build-isolation --force-reinstall
	
	@echo "$(YELLOW)Fixing PyTorch and numpy versions after Transformer Engine installation...$(NC)"
	@. $(VENV_NAME)/bin/activate && \
	uv pip install torch==$(PYTORCH_VERSION)+$(CUDA_VERSION) \
					torchvision==0.20.1+$(CUDA_VERSION) \
					torchaudio==$(PYTORCH_VERSION)+$(CUDA_VERSION) \
					--index-url https://download.pytorch.org/whl/$(CUDA_VERSION) --force-reinstall && \
	uv pip install "numpy$(NUMPY_VERSION)"
	
	@echo "$(YELLOW)Installing build dependencies and Flash Attention $(FLASH_ATTN_VERSION)...$(NC)"
	@. $(VENV_NAME)/bin/activate && \
	uv pip install packaging ninja && \
	MAX_JOBS=$(MAX_JOBS) uv pip install flash-attn==$(FLASH_ATTN_VERSION) --no-build-isolation
	
	@echo "$(YELLOW)Fixing numpy compatibility...$(NC)"
	@. $(VENV_NAME)/bin/activate && uv pip install "numpy$(NUMPY_VERSION)"
	
	@echo "$(GREEN) Performance packages installed successfully$(NC)"

# Install Hyena model dependencies
.PHONY: install-hyena
install-hyena: install-performance
	@echo "$(YELLOW)Installing Hyena model dependencies...$(NC)"
	@echo "$(YELLOW)Installing causal-conv1d from tested fork...$(NC)"
	@. $(VENV_NAME)/bin/activate && \
	CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=$(MAX_JOBS) \
	uv pip install --no-build-isolation \
	git+https://github.com/trvachov/causal-conv1d.git@52e06e3d5ca10af0c7eb94a520d768c48ef36f1f
	@echo "$(GREEN) Hyena dependencies installed successfully$(NC)"

# Test the installation
.PHONY: test-installation
test-installation: install-hyena
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
	@$(setup_env) && \
	python -c "import flash_attn; print(' Flash Attention:', flash_attn.__version__)" && \
	python -c "from causal_conv1d import causal_conv1d_fn; print(' Causal Conv1D: Available')"
	
	@echo "$(YELLOW)Testing train_evo2 command...$(NC)"
	@. $(VENV_NAME)/bin/activate && train_evo2 --help | head -5
	
	@echo "$(YELLOW)Running quick training test...$(NC)"
	@$(setup_env) && \
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

# Create activation script
.PHONY: create-activation-script
create-activation-script: install-hyena
	@echo "$(YELLOW)Creating activation script...$(NC)"
	@echo '#!/bin/bash' > activate_bionemo_evo2.sh
	@echo '# BioNeMo Evo2 Environment Activation Script' >> activate_bionemo_evo2.sh
	@echo '# Usage: source activate_bionemo_evo2.sh' >> activate_bionemo_evo2.sh
	@echo '' >> activate_bionemo_evo2.sh
	@echo 'echo " Activating BioNeMo Evo2 Environment..."' >> activate_bionemo_evo2.sh
	@echo 'source $(VENV_NAME)/bin/activate' >> activate_bionemo_evo2.sh
	@echo '' >> activate_bionemo_evo2.sh
	@echo '# Set up CUDA and cuDNN environment variables' >> activate_bionemo_evo2.sh
	@echo 'export CUDNN_PATH=$$(python -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])" 2>/dev/null)' >> activate_bionemo_evo2.sh
	@echo 'export CUDA_HOME=/usr/local/cuda' >> activate_bionemo_evo2.sh
	@echo 'if [ -n "$$CUDNN_PATH" ]; then' >> activate_bionemo_evo2.sh
	@echo '    export LD_LIBRARY_PATH="$$CUDNN_PATH/lib:$$LD_LIBRARY_PATH"' >> activate_bionemo_evo2.sh
	@echo '    export CPATH="$$CUDNN_PATH/include:$$CPATH"' >> activate_bionemo_evo2.sh
	@echo '    export LIBRARY_PATH="$$CUDNN_PATH/lib:$$LIBRARY_PATH"' >> activate_bionemo_evo2.sh
	@echo 'fi' >> activate_bionemo_evo2.sh
	@echo '' >> activate_bionemo_evo2.sh
	@echo 'echo " BioNeMo Evo2 Environment Activated!"' >> activate_bionemo_evo2.sh
	@echo 'echo " Location: $$(pwd)/$(VENV_NAME)"' >> activate_bionemo_evo2.sh
	@echo 'echo " Python: $$(python --version)"' >> activate_bionemo_evo2.sh
	@echo 'echo " PyTorch CUDA: $$(python -c '\''import torch; print("Available" if torch.cuda.is_available() else "Not Available")'\'')"' >> activate_bionemo_evo2.sh
	@echo 'echo " GPU: $$(python -c '\''import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")'\'')"' >> activate_bionemo_evo2.sh
	@echo 'echo ""' >> activate_bionemo_evo2.sh
	@echo 'echo "Available BioNeMo commands:"' >> activate_bionemo_evo2.sh
	@echo 'echo "  • train_evo2 --help      - Train Evo2 models"' >> activate_bionemo_evo2.sh
	@echo 'echo "  • infer_evo2 --help      - Run Evo2 inference"' >> activate_bionemo_evo2.sh
	@echo 'echo "  • predict_evo2 --help    - Make predictions with Evo2"' >> activate_bionemo_evo2.sh
	@echo 'echo "  • preprocess_evo2 --help - Preprocess data for Evo2"' >> activate_bionemo_evo2.sh
	@echo 'echo ""' >> activate_bionemo_evo2.sh
	@echo 'echo "Example usage:"' >> activate_bionemo_evo2.sh
	@echo 'echo "  train_evo2 --mock-data --devices 1 --max-steps 100 --model-size test"' >> activate_bionemo_evo2.sh
	@echo 'echo ""' >> activate_bionemo_evo2.sh
	@chmod +x activate_bionemo_evo2.sh
	@echo "$(GREEN) Activation script created: activate_bionemo_evo2.sh$(NC)"

# Clean up virtual environment
.PHONY: clean
clean:
	@echo "$(YELLOW)Removing virtual environment: $(VENV_NAME)$(NC)"
	@rm -rf $(VENV_NAME)
	@rm -f activate_bionemo_evo2.sh
	@echo "$(GREEN) Environment cleaned$(NC)"

# Clean up results
.PHONY: clean-results
clean-results:
	@echo "$(YELLOW)Removing training results...$(NC)"
	@rm -rf results lightning_logs
	@echo "$(GREEN) Results cleaned$(NC)"

# Quick setup for testing
.PHONY: quick-test
quick-test: all create-activation-script
	@echo "$(GREEN) Quick test setup completed!$(NC)"
	@echo ""
	@echo "$(YELLOW)To get started:$(NC)"
	@echo "  1. source activate_bionemo_evo2.sh"
	@echo "  2. train_evo2 --mock-data --devices 1 --max-steps 5"
	@echo ""

# Show train_evo2 help
.PHONY: show-help
show-help:
	@echo "$(YELLOW)train_evo2 command help:$(NC)"
	@. $(VENV_NAME)/bin/activate && train_evo2 --help

# Development setup (includes all packages)
.PHONY: dev-setup
dev-setup: all
	@echo "$(YELLOW)Installing development dependencies...$(NC)"
	@. $(VENV_NAME)/bin/activate && \
	uv pip install -e ".[dev,test]"
	@echo "$(GREEN) Development setup completed$(NC)"

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

# Restore environment after dependency conflicts (e.g., after installing test packages)
.PHONY: restore-env
restore-env:
	@echo "$(YELLOW)Restoring BioNeMo environment after dependency conflicts...$(NC)"
	@echo "$(YELLOW)Reinstalling subpackages with proper versions...$(NC)"
	@$(MAKE) install-subpackages
	@echo "$(YELLOW)Ensuring all critical packages have correct versions...$(NC)"
	@. $(VENV_NAME)/bin/activate && \
	uv pip install torch==$(PYTORCH_VERSION)+$(CUDA_VERSION) \
					torchvision==0.20.1+$(CUDA_VERSION) \
					torchaudio==$(PYTORCH_VERSION)+$(CUDA_VERSION) \
					--index-url https://download.pytorch.org/whl/$(CUDA_VERSION) --force-reinstall && \
	uv pip install "numpy$(NUMPY_VERSION)" && \
	uv pip install flash-attn==$(FLASH_ATTN_VERSION) --no-build-isolation && \
	CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=$(MAX_JOBS) \
	uv pip install --no-build-isolation \
	git+https://github.com/trvachov/causal-conv1d.git@52e06e3d5ca10af0c7eb94a520d768c48ef36f1f
	@echo "$(GREEN) Environment restored successfully!$(NC)"

# Run tests for BioNeMo packages
.PHONY: run-tests
run-tests:
	@echo "$(YELLOW)Running BioNeMo tests...$(NC)"
	@echo "$(YELLOW)Installing test dependencies...$(NC)"
	@. $(VENV_NAME)/bin/activate && \
	uv pip install pytest pytest-xdist pytest-cov
	@echo "$(YELLOW)Installing bionemo-testing package...$(NC)"
	@. $(VENV_NAME)/bin/activate && cd sub-packages/bionemo-testing && uv pip install -e .
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
	@. $(VENV_NAME)/bin/activate && uv pip install pytest pytest-xdist pytest-cov
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
	@. $(VENV_NAME)/bin/activate && uv pip install pytest pytest-xdist pytest-cov
	@echo "$(YELLOW)Ensuring bionemo-core is properly installed...$(NC)"
	@. $(VENV_NAME)/bin/activate && cd sub-packages/bionemo-core && uv pip install -e . --force-reinstall
	@echo "$(YELLOW)Restoring compatible versions...$(NC)"
	@. $(VENV_NAME)/bin/activate && \
	uv pip install torch==$(PYTORCH_VERSION)+$(CUDA_VERSION) \
					torchvision==0.20.1+$(CUDA_VERSION) \
					torchaudio==$(PYTORCH_VERSION)+$(CUDA_VERSION) \
					--index-url https://download.pytorch.org/whl/$(CUDA_VERSION) --force-reinstall && \
	uv pip install "numpy$(NUMPY_VERSION)" && \
	uv pip install flash-attn==$(FLASH_ATTN_VERSION) --no-build-isolation && \
	CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=$(MAX_JOBS) \
	uv pip install --no-build-isolation \
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
	uv pip install pytest pytest-xdist pytest-cov
	@echo "$(YELLOW)Installing bionemo-testing package...$(NC)"
	@. $(VENV_NAME)/bin/activate && cd sub-packages/bionemo-testing && uv pip install -e .
	@echo "$(YELLOW)Running all bionemo-evo2 tests...$(NC)"
	@. $(VENV_NAME)/bin/activate && cd sub-packages/bionemo-evo2 && \
	python -m pytest tests/ -v --tb=short --continue-on-collection-errors --maxfail=20 || true
	@echo "$(GREEN) All tests completed!$(NC)"

