.PHONY: test build clean

# Install package
install:
	pip install poetry==1.8.4
	poetry install --with dev

# Run PettingZoo example
pz:
	poetry run python3 -m examples.pettingzoo_example

# Run Gymnasium example
gym:
	poetry run python3 -m examples.gymnasium_example

remote:
	poetry run python3 -m examples.client_example
# Create new replay and run it
make n_replay:
	poetry run python3 -m examples.record_replay_example
	poetry run python3 -m examples.show_replay_example

# Run existing replay
replay:
	poetry run python3 -m examples.show_replay_example

###################
# Developer tools #
###################

test_performance:
	poetry run python3 -m tests.parallel_api_check

test:
	poetry run pytest

pc:
	poetry run pre-commit run --all-files

build:
	poetry build

clean:
	rm -rf build dist *.egg-info
