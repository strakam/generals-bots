import os
from generals.replay import Replay
from generals.agent import RandomAgent
from generals.map import Mapper
from generals.env import pz_generals

def test_replays():
    # run N games, store their replay, then load the replay and compare game states
    for _ in range(3):
        agents = [RandomAgent(name="A"), RandomAgent(name="B")]
        mapper = Mapper(
            grid_size=3,
            mountain_density=0.2,
            city_density=0.05,
            general_positions=[(0, 0), (2, 2)],
        )

        env = pz_generals(mapper, agents, render_mode="none")
        observations, info = env.reset(options={"replay_file": "replay_test"})

        agents = {agent.name: agent for agent in agents}
        while not env.game.is_done():
            actions = {}
            for agent in env.agents:
                actions[agent] = agents[agent].play(observations[agent])
            observations, rewards, terminated, truncated, info = env.step(actions)
        replay_before = env.replay
        replay_after = Replay.load("replay_test")

        assert replay_before.name == replay_after.name
        assert replay_before.map == replay_after.map
        # compare agent data dicts
        for before, after in zip(replay_before.agent_data, replay_after.agent_data):
            assert before == after
            
        for before, after in zip(replay_before.game_states, replay_after.game_states):
            # Check if they have the same channels
            before_keys = before.keys()
            after_keys = after.keys()
            assert before_keys == after_keys

            # For each channel check if they are the same
            for key in before_keys:
                assert (before[key] == after[key]).all()
    os.remove("replay_test.pkl")