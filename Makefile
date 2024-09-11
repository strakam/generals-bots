replay:
	python3 -m examples.replay

run:
	python3 -m examples.run_environment


t_all:
	pytest tests/test_logic.py
	pytest tests/test_map.py
	python3 -m tests.parallel_api_test

t_game:
	pytest tests/test_logic.py

t_map:
	pytest tests/test_map.py

t_parallel:
	python3 -m tests.parallel_api_test
