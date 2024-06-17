run:
	python3 -m tests.test_run

test_all:
	pytest tests/test_logic.py
	pytest tests/test_map.py
	python3 -m tests.parallel_api_test

test_game:
	pytest tests/test_logic.py

test_map:
	pytest tests/test_map.py

test_parallel:
	python3 -m tests.parallel_api_test
