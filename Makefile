.PHONY: check delete zip map train

check:
	uv run ruff format .
	uv run ruff check --fix utils tests train.py model.py
	uv run mypy utils tests train.py model.py
	uv run pytest --cov
	uv run xenon --max-absolute B --max-average A --max-modules B utils/ tests/ train.py model.py

delete:
	find . -type d \( \
	  -name '__pycache__' -o \
	  -name '.pytest_cache' -o \
	  -name '.ruff_cache' -o \
	  -name '.mypy_cache' -o \
	  -name '.hypothesis_cache' -o \
	  -name '.ipynb_checkpoints' \
	\) -prune -exec rm -rf -- {} +
	find . -type f -name '.DS_Store' -delete

zip:
	zip -r cursor.zip .cursor

map:
	uv run .cursor/skills/map-architecture/generate_architecture_map.py --output docs/architecture_map.md
train:
	env -u CK_LOGGING \
		MIOPEN_ENABLE_LOGGING=0 \
		MIOPEN_ENABLE_LOGGING_CMD=0 \
		MIOPEN_LOG_LEVEL=1 \
		MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F2X3=0 \
		MIOPEN_DEBUG_AMD_WINOGRAD_FURY_RXS_F3X2=0 \
		MIOPEN_FIND_MODE=2 \
		HIPBLASLT_LOG_LEVEL=0 \
		HIPBLASLT_LOG_MASK=0 \
		ROCBLAS_LAYER=0 \
		uv run python train.py
