run:
	python3 -m tests.test_run

test_all:
	pytest tests/test_logic.py
	python3 -m tests.parallel_api_test

test_logic:
	pytest tests/test_logic.py

test_parallel:
	python3 -m tests.parallel_api_test
