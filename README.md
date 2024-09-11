<div align="center">

[<img src="https://github.com/strakam/Generals-Zoo/blob/master/generals/images/test.png?raw=true" alt="Generals-Zoo logo" width="500"/>](https://github.com/strakam/Generals-Zoo)

**Machine Learning Friendly Implementation of the Generals.io Game**
 
</div>

[generals.io](https://generals.io/) is a real-time strategy game where players compete to conquer their opponents' generals on a 2D grid. While the goal is simple — capture the enemy general — the gameplay involves a lot of depth. Players need to employ strategic planning, deception, and manage both micro and macro mechanics throughout the game. The combination of these elements makes the game highly engaging and complex.

From a bot development perspective, generals.io has interesting characteristics. It’s easy to control but challenging due to its partially observable environment, real-time dynamics, long sequences of actions, and a large action space.

While the game has an active player base, with tournaments being held regularly, the bot landscape remains quite limited. Currently, there’s only one "rule-based" bot that can consistently beat human players. This raises an intriguing question: can Machine Learning (ML) agents, powered by recent advances in (Deep) Reinforcement Learning, outperform the current state-of-the-art bot?

One challenge to answering this question is that the official botting API makes it tough to implement ML-based agents, particularly for developers who want to integrate modern ML frameworks. To solve this, our project aims to make bot development more accessible by creating a numpy-powered environment that is fully compatible with the [PettingZoo](https://pettingzoo.farama.org/) API standard. This simplifies the development process, making it as easy as implementing a single function.


## Installation/Setup
TODO

## Usage example
TODO : map from string
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
Maps are encoded using these symbols:
- `.` for passable terrain
- `#` for non-passable terrain
- `A,B` are positions of generals
- digits `0-9` represent cost of cities calculated as 40 + digit

## Replay analysis
We can store replays and then analyze them.
### Storing a replay
```python
from generals.env import generals_v0
from generals.config import GameConfig

game_config = GameConfig()
options = {"replay_file": "replay_001"}
env.reset(options=options) # encodes the next game into a "replay_001" file
```

### Loading a replay
```python
import generals.utils
import generals.agents
from generals.config import GameConfig


# Create agents - their names are then called for actions
agents = {
    "Red": generals.agents.RandomAgent("Red"),
    "Blue": generals.agents.RandomAgent("Blue")
}

testik = GameConfig(
    grid_size=16,
    agent_names=list(agents.keys()),
    replay_file="replay_001"
)

# Run from replay - user can analyze the game and try different runs
generals.utils.run(testik, agents) # Pass whole agents as second parameter
```
### Replay Controls
- `q` — quit/close the game
- `←/→` — increase/decrease the game speed
- `h/l` — to control replay frames
- `spacebar` — to pause
- `Mouse` click on the player's row — toggle the FOV (Field Of View) of the given player
## POMDP - Observations, Actions and Rewards
### Observation
An observation for one player is a dictionary of 8 key/value pairs. Each value is a 2D `np.array` containing information for each cell.
Values are (binary) masked so that only information about cells that an agent can see can be non-zero.
|Key|Shape|Description|
|---|---|---|
|`army`| `(N,N,1)` | Number of units in a cell regardless of owner|
|`general`| `(N,N,1)` | Mask of cells that are visible to the agent|
|`city`| `(N,N,1)` | Mask saying whether a city is in a cell|
|`ownership`| `(N,N,1)` | Mask indicating cells controlled by the agent|
|`ownership_opponent`| `(N,N,1)` | Mask indicating cells owned by the opponent|
|`ownership_neutral`| `(N,N,1)` | Mask indicating cells that are not owned by agents|
|`structure`| `(N,N,1)` | Mask indicating whether cells contain cities or mountains, even out of FoV|
|`action_mask`| `(N,N,4)` | Mask where `[i,j,k]` indicates whether you can move from a cell `[i,j]` to direction `k` where directions are in order (UP, DOWN, LEFT, RIGHT)|
   
### Action
Actions are `np.array` with shape `(3,)` which indicate that you want to move units from cell `[i,j]` in a direction `k`.

### Reward
TODO
## TODOs:
- Replays need improvements
- Extend action space to sending half of units to another square
- Add human control to play against
