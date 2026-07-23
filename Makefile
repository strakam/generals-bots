.PHONY: test bench match

test:
	pytest tests

bench:
	python bench.py

# Local match between two reference bots under the competition ruleset
match:
	python competition/matchup.py --mode competition
