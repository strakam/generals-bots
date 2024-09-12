replay:
	python3 -m examples.replay

run:
	python3 -m examples.run_environment

t:
	pytest tests/test_game.py
	pytest tests/test_utils.py
	python3 -m tests.parallel_api_test
