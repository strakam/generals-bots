<div align="center">

[<img src="https://github.com/strakam/Generals-Zoo/blob/master/generals/images/test.png?raw=true" alt="Generals-Zoo logo" width="500"/>](https://github.com/strakam/Generals-Zoo)

**Machine Learning Friendly Implementation of the Generals.io Game**
 
</div>

[generals.io](https://generals.io/) is a real-time strategy game where players compete to conquer their opponents' generals on a 2D grid. While the goal is simple—capture the enemy general—the gameplay involves a lot of depth. Players need to employ strategic planning, deception, and manage both micro and macro mechanics throughout the game. The combination of these elements makes the game highly engaging and complex.

From a bot development perspective, generals.io has interesting characteristics. It’s easy to control but challenging due to its partially observable environment, real-time dynamics, long sequences of actions, and a large action space.

While the game has an active player base, with tournaments being held regularly, the bot landscape remains quite limited. Currently, there’s only one "rule-based" bot that can consistently beat human players. This raises an intriguing question: can Machine Learning (ML) agents, powered by recent advances in (Deep) Reinforcement Learning, outperform the current state-of-the-art bot?

One challenge to answering this question is that the official botting API makes it tough to implement ML-based agents, particularly for developers who want to integrate modern ML frameworks. To solve this, our project aims to make bot development more accessible by creating a numpy-powered environment that is fully compatible with the [PettingZoo](https://pettingzoo.farama.org/) API standard. This simplifies the development process, making it as easy as implementing a single function.


## Installation/Setup
TODO

## Usage example

```python
from generals.env import generals_v0
from generals.agents import RandomAgent
from generals.config import GameConfig

# Initialize agents - their names are then called for actions
agents = {
    "red": RandomAgent("red"),
    "blue": RandomAgent("blue")
}

agent_names = list(agents.keys())

game_config = GameConfig(
    grid_size=4,
    agent_names=agent_names
)

# Create environment
env = generals_v0(game_config, render_mode="none")
observations, info = env.reset()

while not env.game.is_done():
    actions = {}
    for agent in env.agents:
        # Ask agent for action
        actions[agent] = agents[agent].play(observations[agent])
    # All agents perform their actions
    observations, rewards, terminated, truncated, info = env.step(actions)
```

## Replay analysis

## Customization
The environment can be customized via `GridConfig` class or by creating a custom map.

### Random maps
```python
from generals.env import generals_v0
from generals.config import GameConfig

game_config = GameConfig(
    grid_size=16,                         # Edge length of the square grid
    mountain_density=0.2                  # Probability of mountain in a cell
    town_density=0.05                     # Probability of town in a cell
    general_positions=[(0,3),(5,7)]       # Positions of generals (i, j)
    agent_names=['Human.exe','Agent007']  # Names of the agents that will be called to play the game
)

# Create environment
env = generals_v0(game_config, render_mode="none")
observations, info = env.reset()
```

### Custom maps
Maps can be described by strings. We can either load them directly from a string or from a file.

#### Map from string
```python
from generals.env import generals_v0
from generals.config import GameConfig

game_config = GameConfig(
    agent_names=['Human.exe','Agent007']  # Names of the agents that will be called to play the game
)
map = """
...#
#..A
#..#
.#.B
"""

env.reset(map=map) # Here map related settings from game_config are overridden
```

TODO : map from string

## Controls
- `q` — quit/close the game
- `←/→` — increase/decrease the game speed
- `h/l` - to control replay frames
- `spacebar` - to pause
- `Mouse` click on the player's row — toggle the FOV (Field Of View) of the given player
