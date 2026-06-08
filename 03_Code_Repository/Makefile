PYTHON ?= python3

.PHONY: reproduce reproduce-final ablation report-assets test all

all: reproduce test

reproduce: reproduce-final ablation report-assets

reproduce-final:
	PYTHONPATH=src $(PYTHON) -m c2o_strategy.final_strategy --data-dir data --output-dir outputs --cutoff 2024-12-31 --development-cutoff 2024-12-31 --mode all

ablation:
	PYTHONPATH=src $(PYTHON) -m c2o_strategy.final_strategy --data-dir data --output-dir outputs --cutoff 2024-12-31 --development-cutoff 2024-12-31 --mode ablation

report-assets:
	PYTHONPATH=src $(PYTHON) -m c2o_strategy.final_strategy --data-dir data --output-dir outputs --cutoff 2024-12-31 --development-cutoff 2024-12-31 --mode audits

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q
