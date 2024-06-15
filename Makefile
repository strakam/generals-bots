run:
	python3 -m tests.test_run

test:
	pytest tests/test_logic.py
	python3 -m tests.parallel_api_test

