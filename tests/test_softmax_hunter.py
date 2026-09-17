"""The hosted Hunter must see and choose exactly what the original agent does."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from generals.agents.hunter_agent import HunterAgent
from generals.core.game import get_observation
from integrations.softmax.engine import Match
from integrations.softmax.hunter_player import observation_from_message


@pytest.mark.parametrize("slot", [0, 1])
def test_hunter_wire_observation_and_actions_match_engine(slot):
    match = Match(7)
    agent, key = HunterAgent(), jax.random.PRNGKey(0)
    # Cover changing army/land totals and newly visible terrain on both seats.
    for _ in range(25):
        expected = get_observation(match.state, slot)
        received = observation_from_message(match.observation(slot))
        for field in expected._fields:
            np.testing.assert_array_equal(getattr(received, field), getattr(expected, field), err_msg=field)
        original = agent.act(expected, key)
        transported = agent.act(received, key)
        np.testing.assert_array_equal(transported, original)
        actions = jnp.stack([agent.act(get_observation(match.state, i), key) for i in range(2)])
        match.state, _ = match.transition(match.state, actions)
