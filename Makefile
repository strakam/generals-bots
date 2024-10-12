.PHONY: test build clean


# Run PettingZoo example
pz:
	python3 -m examples.pettingzoo_example

# Run Gymnasium example
gym:
	python3 -m examples.gymnasium_example

# Create new replay and run it
make n_replay:
	python3 -m examples.record_replay_example
	python3 -m examples.show_replay_example

# Run existing replay
replay:
	python3 -m examples.show_replay_example

###################
# Developer tools #
###################

test_performance:
	python3 -m tests.parallel_api_check

test:
	pytest

build:
	python setup.py sdist bdist_wheel

clean:
	rm -rf build dist *.egg-info
