run:
	python3 -m tests.test_run

human:
	python3 -m tests.test_human

t_replay:
	python3 -m tests.test_replay

control:
	python3 -m generals.wrappers.human_control

scratch:
	python3 -m tests.scratch
	cat rb

# t_something is short for test_something
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
