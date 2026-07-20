DATA_DIR ?= path/to/gamelytics_data
BOOTSTRAP_ITERATIONS ?= 5000
SEED ?= 42

.PHONY: all test ab retention

all: test ab retention

test:
	python -m pytest tests -q

ab:
	python scripts/run_ab_analysis.py --data-dir "$(DATA_DIR)" --bootstrap-iterations $(BOOTSTRAP_ITERATIONS) --permutation-iterations $(BOOTSTRAP_ITERATIONS) --seed $(SEED)

retention:
	python scripts/run_retention_analysis.py --data-dir "$(DATA_DIR)"
