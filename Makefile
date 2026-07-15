.PHONY: test run lint eval

test:
	pytest

run:
	python scripts/run_eval.py

eval:
	python scripts/run_eval.py --evaluate-only

lint:
	ruff check .
	ruff format --check .
