from generals.utils import map_from_generator, validate_map, map_from_string, load_replay
from generals.env import pz_generals
from generals.config import GameConfig
from generals.agents import RandomAgent
from copy import deepcopy


def test_validate_map():
    map = """
.....
.A##2
...2.
..22.
...B.
    """
    map = map_from_string(map)
    assert validate_map(map)

    map = """
.....
.A##2
##.2.
..###
...B.
    """
    map = map_from_string(map)
    assert not validate_map(map)

    map = """
.....
.A##2
##.2.
..2##
...B.
    """
    map = map_from_string(map)
    assert validate_map(map)

    map = """
..#..
.A##2
##.2.
..2##
...B.
    """
    map = map_from_string(map)
    assert not validate_map(map)

    map = """
.....
BA2#2
##.2.
..2##
.....
    """
    map = map_from_string(map)
    assert validate_map(map)

    map = """
...#.
#A2#2
##.#B
..2##
.....
    """
    map = map_from_string(map)
    assert not validate_map(map)

    map = """
...#.
A#2#2
##.#B
..2#.
.....
    """
    map = map_from_string(map)
    assert validate_map(map)


def test_generate_map():
    grid_size, mountain_density, city_density = 16, 0.1, 0.1
    for _ in range(5):
        map = map_from_generator(grid_size, mountain_density, city_density)
        assert validate_map(map)  # map has to be valid
        assert map.shape == (grid_size, grid_size)

    grid_size, mountain_density, town_density = 10, 0.2, 0.2
    for _ in range(5):
        map = map_from_generator(grid_size, mountain_density, town_density)
        assert validate_map(map)  # map has to be valid
        assert map.shape == (grid_size, grid_size)


def test_replays():
    # run N games, store their replay, then load the replay and compare game states
    for _ in range(3):
        agents = [RandomAgent(name="A"), RandomAgent(name="B")]
        game_states_before, game_states_after = [], []

        game_config = GameConfig(
            grid_size=4,
            mountain_density=0.2,
            city_density=0.05,
            agents=agents,
        )
        env = pz_generals(game_config, render_mode=None)
        observations, info = env.reset(options={"replay_file": "test"})

        agents = {agent.name: agent for agent in agents}
        game_states_before.append(deepcopy(env.game.channels))
        while not env.game.is_done():
            actions = {}
            for agent in env.agents:
                actions[agent] = agents[agent].play(observations[agent])
            observations, rewards, terminated, truncated, info = env.step(actions)
            game_states_after.append(deepcopy(env.game.channels))
        agents_after, map, game_states_after = load_replay("test")
        assert list(agents.keys()) == agents_after
        for before, after in zip(game_states_before, game_states_after):
            # Check if they have the same channels
            before_keys = before.keys()
            after_keys = after.keys()
            assert before_keys == after_keys

            # For each channel check if they are the same
            for key in before_keys:
                assert (before[key] == after[key]).all()
