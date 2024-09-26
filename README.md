<div align="center">

![Gameplay GIF](https://raw.githubusercontent.com/strakam/Generals-RL/master/generals/images/gameplay.gif)

## **Generals.io RL**

[![CodeQL](https://github.com/strakam/Generals-RL/actions/workflows/codeql.yml/badge.svg)](https://github.com/strakam/Generals-RL/actions/workflows/codeql.yml)
[![CI](https://github.com/strakam/Generals-RL/actions/workflows/tests.yml/badge.svg)](https://github.com/strakam/Generals-RL/actions/workflows/tests.yml)




[Installation](#-installation) • [Customization](#-customization) • [Environment](#-environment) • [Getting Started](#-getting-started) 

</div>

[Generals.io](https://generals.io/) is a real-time strategy game where players compete to conquer their opponents' generals on a 2D grid. While the goal is simple — capture the enemy general — the gameplay involves a lot of depth. Players need to employ strategic planning, deception, and manage both micro and macro mechanics throughout the game. The combination of these elements makes the game highly engaging and complex.

This repository aims to make bot development more accessible, especially for Machine Learning based agents.

Highlights:
* 🚀 Fast & Lightweight simulator powered by `numpy` (thousands of steps per second)
* 🤝 Compatibility with Reinforcement-Learning API standards 🤸[Gymnasium](https://gymnasium.farama.org/) and 🦁[PettingZoo](https://pettingzoo.farama.org/)
* 🔧 Easy customization of environments
* 🔬 Analysis tools such as replays

<br>

Generals.io has several interesting properties:
* 👀 Partial observability
* 🏃‍♂️ Long action sequences and large action spaces
* 🧠 Requires strategical planning
* ⏱️ Real-time gameplay 


## 📦 Installation
Stable release version is available through pip:
```bash
pip install generals
```
Alternatively, you can install latest version via git
```bash
git clone https://github.com/strakam/Generals-RL
cd Generals-RL
pip install -e .
```

## Usage Example (🦁 PettingZoo)
```python
from generals.env import pz_generals
from generals.agent import ExpanderAgent, RandomAgent
from generals.map import Mapper

# Initialize agents
agents = [RandomAgent(), ExpanderAgent()]
mapper = Mapper(
    grid_size=16,
    mountain_density=0.2,
    city_density=0.05,
    general_positions=[(4, 12), (12, 4)],
)

# Create environment
env = pz_generals(mapper, agents, render_mode="human")  # render_mode {None, "human"}
observations, info = env.reset(options={"replay_file": "replay"})

# How fast we want rendering to be
actions_per_second = 6
agents = {
    agent.name: agent for agent in agents
}  # Create a dictionary of agents - their names are then called for actions

while not env.game.is_done():
    actions = {}
    for agent in env.agents:
        # Ask agent for action
        actions[agent] = agents[agent].play(observations[agent])
    # All agents perform their actions
    observations, rewards, terminated, truncated, info = env.step(actions)
    env.render(tick_rate=actions_per_second)
```

## Usage example (🤸 Gymnasium)
```python
from generals.env import gym_generals
from generals.agent import RandomAgent, ExpanderAgent
from generals.map import Mapper

# Initialize agents
agent = RandomAgent()
npc = ExpanderAgent()

agents = [agent, npc]  # First is player, second is NPC

mapper = Mapper(
    grid_size=16,
    mountain_density=0.2,
    city_density=0.05,
    general_positions=[(4, 12), (12, 4)],
)

# Create environment
env = gym_generals(mapper, agents, render_mode="human")  # render_mode {None, "human"}
observation, info = env.reset()

# How fast we want rendering to be
actions_per_second = 6
agents = {
    agent.name: agent for agent in agents
} # Create a dictionary of agents - their names are then called for actions

done = False

while not done:
    action = agent.play(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render(tick_rate=actions_per_second)
```

## 🎨 Customization
The environment can be customized via `GridConfig` class or by creating a custom map.

### 🗺️ Random maps
```python
from generals.env import pz_generals
from generals.config import GameConfig

game_config = GameConfig(
    grid_size=16,                          # Edge length of the square grid
    mountain_density=0.2,                  # Probability of a mountain in a cell
    city_density=0.05,                     # Probability of a city in a cell
    general_positions=[(0,3),(5,7)],       # Positions of generals (i, j)
)

# Create environment
env = pz_generals(game_config)
observations, info = env.reset()
```

### 🗺️ Custom maps
Maps can be described by strings. We can either load them directly from a string or from a file.

```python
from generals.env import pz_generals
from generals.config import GameConfig

game_config = GameConfig()
map = """
.3.#
#..A
#..#
.#.B
"""

env = pz_generals(game_config)
env.reset(map=map) # Here map related settings from game_config are overridden
```
Maps are encoded using these symbols:
- `.` for passable terrain
- `#` for non-passable terrain
- `A,B` are positions of generals
- digits `0-9` represent cost of cities calculated as `(40 + digit)`

## 🔬 Replay Analysis
We can store replays and then analyze them.
### Storing a replay
```python
from generals.env import pz_generals
from generals.config import GameConfig

game_config = GameConfig()
options = {"replay_file": "replay_001"}
env = pz_generals(game_config)
env.reset(options=options) # encodes the next game into a "replay_001" file
```

### Loading a replay
The following code loads and executes replay named `replay_001`:
```python
import generals.utils

generals.utils.run_replay("replay_001")
```
### 🕹️ Replay controls
- `q` — quit/close the replay
- `←/→` — increase/decrease the replay speed
- `h/l` — move backward/forward by one frame in the replay
- `spacebar` — toggle play/pause
- `mouse` click on the player's row — toggle the FoV (Field Of View) of the given player

## 🌍 Environment
### 🔭 Observation
An observation for one agent is a dictionary of 13 key/value pairs. Each key/value pair contains information about part of the game that is accessible to the agent.
| Key                  | Shape/Type| Description                                                                                                                                    |
| ---                  | ---       | ---                                                                                                                                            |
| `army`               | `(N,N,1)` | Number of units in a cell regardless of owner                                                                                                  |
| `general`            | `(N,N,1)` | Mask of cells that are visible to the agent                                                                                                    |
| `city`               | `(N,N,1)` | Mask saying whether a city is in a cell                                                                                                        |
| `visibile_cells`     | `(N,N,1)` | Mask indicating cells that are visible to the agent                                                                                            |
| `owned_cells`        | `(N,N,1)` | Mask indicating cells controlled by the agent                                                                                                  |
| `opponent_cells`     | `(N,N,1)` | Mask indicating cells owned by the opponent                                                                                                    |
| `neutral_cells`      | `(N,N,1)` | Mask indicating cells that are not owned by agents                                                                                             |
| `structure`          | `(N,N,1)` | Mask indicating whether cells contain cities or mountains, even out of FoV                                                                     |
| `action_mask`        | `(N,N,4)` | Mask where `[i,j,d]` indicates whether you can move from a cell `[i,j]` to direction `d` where directions are in order (UP, DOWN, LEFT, RIGHT) |
| `owned_land_count`   | `Int`     | Int representing number of cells an agent owns                                                                                                 |
| `owned_army_count`   | `Int`     | Int representing total number of units of an agent over all cells                                                                              |
| `opponent_land_count`| `Int`     | Int representing number of cells owned by the opponent                                                                                         |
| `opponent_army_count`| `Int`     | Int representing total number of units owned by the opponent                                                                                   |
| `is_winner`          | `Bool`    | Bool representing whether an agent won                                                                                                         |
| `timestep`           | `Int`     | Timestep                                                                                                                                       |
   
### ⚡ Action
Action is an `np.array([pass,i,j,d,split])`:
- Value of `pass` indicates whether you want to `1 (pass)` or `0 (play)`.
- Indices `i,j` say that you want to move from cell with index `[i,j]`.
- Value of `d` is a direction of the movement: `0 (up)`, `1 (down)`, `2 (left)`, `3 (right)`
- Value of `split` says whether you want to split units. Value `1 (split)` sends half of units and value `0 (no split)` sends all possible units to the next cell.

### 🎁 Reward
It is possible to implement custom reward function. The default is `1` for winner and `-1` for loser, otherwise `0`.
```python
def custom_reward_fn(observations):
    # Give agent a reward based on the number of cells they own
    return {
        agent: observations[agent]["owned_land_count"]
        for agent in observations.keys()
    }

env = generals_v0(reward_fn=custom_reward_fn)
observations, info = env.reset()
```
## 🚀 Getting Started
Creating your first agent is very simple. Start by subclassing an `Agent` class just like `RandomAgent` [here](./generals/agents.py).
- Every agent must have a name as it is his ID by which he is called for actions.
- Every agent must implement `play(observation)` function that takes in `observation` and returns an action as described above.
- You can simply follow examples to make your bot running.
- When creating an environment, you can choose out of two `render_modes`:
     - `none` that omits rendering and is suitable for training,
     - `human` where you can see the game roll out.

## 🛠️ Coming Soon
- Examples and baselines using RL
- Add human control to play against
  
Requests for useful features and additions are welcome 🤗.
