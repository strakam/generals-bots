.PHONY: test build clean


# Run PettingZoo example
pz:
	python3 -m examples.pettingzoo_example

# Run Gymnasium example
gym:
	python3 -m examples.gymnasium_example

# Create new replay and run it
make n_replay:
	python3 -m examples.dummy
	python3 -m examples.replay

# Run existing replay
replay:
	python3 -m examples.replay

###################
# Developer tools #
###################
t:
	pytest tests/test_game.py
	pytest tests/test_utils.py
	python3 tests/sb3_check.py
	python3 -m tests.parallel_api_test

build:
	python setup.py sdist bdist_wheel

clean:
	rm -rf build dist
