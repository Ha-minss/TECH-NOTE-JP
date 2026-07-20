PYTHON ?= python
PROJECT_ROOT ?= .
ZIP_PATH ?= $(USERPROFILE)/Downloads/mobile-game-ltv-forecasting-challenge.zip

.PHONY: all validate features train predict business test experiments clean

all: validate features train predict business

validate:
	$(PYTHON) src/pipeline/validate_raw_data.py --project-root $(PROJECT_ROOT) --zip-path $(ZIP_PATH)

features:
	$(PYTHON) src/pipeline/build_features.py --output-root $(PROJECT_ROOT) --zip-path $(ZIP_PATH)
	$(PYTHON) src/pipeline/validate_model_input.py --feature-root $(PROJECT_ROOT)

train:
	$(PYTHON) src/pipeline/train_final_model.py --project-root $(PROJECT_ROOT) --zip-path $(ZIP_PATH)

predict:
	$(PYTHON) src/pipeline/predict_submission.py --project-root $(PROJECT_ROOT) --zip-path $(ZIP_PATH)

business:
	$(PYTHON) src/pipeline/build_business_report.py --project-root $(PROJECT_ROOT) --zip-path $(ZIP_PATH)

test:
	$(PYTHON) -m unittest discover tests -v

experiments:
	$(PYTHON) src/experiments/run_experiments.py --project-root $(PROJECT_ROOT) --zip-path $(ZIP_PATH)

clean:
	$(PYTHON) src/pipeline/clean_outputs.py
