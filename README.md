<div align="center">

[<img src="https://github.com/strakam/Generals-Zoo/blob/master/generals/images/test.png?raw=true" alt="Generals-Zoo logo" width="500"/>](https://github.com/strakam/Generals-Zoo)

## **Generals.io RL**
 
</div>

[generals.io](https://generals.io/) is a real-time strategy game where players compete to conquer their opponents' generals on a 2D grid. While the goal is simple — capture the enemy general — the gameplay involves a lot of depth. Players need to employ strategic planning, deception, and manage both micro and macro mechanics throughout the game. The combination of these elements makes the game highly engaging and complex.

This repository aims to make bot development more accessible, especially for Machine Learning based agents.

Highlights:
* 🚀 Fast & Lightweight simulator powered by numpy (thousands of steps per second)
* 🦁 Compatibility with Reinforcement-Learning API standard [PettingZoo](https://pettingzoo.farama.org/)
* 🔧 Easy customization of environments
* 🔬 Analysis tools such as replays

<br>

Generals.io has interesting properties:
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

game_config = GameConfig(
    grid_size=4,
    agent_names=list(agents.keys())
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

## 🎨 Customization
The environment can be customized via `GridConfig` class or by creating a custom map.

### 🗺️ Random maps
```python
from generals.env import generals_v0
from generals.config import GameConfig

game_config = GameConfig(
    grid_size=16,                         # Edge length of the square grid
    mountain_density=0.2                  # Probability of a mountain in a cell
    city_density=0.05                     # Probability of a city in a cell
    general_positions=[(0,3),(5,7)]       # Positions of generals (i, j)
    agent_names=['Human.exe','Agent007']  # Names of the agents that will be called to play the game
)

# Create environment
env = generals_v0(game_config, render_mode="none")
observations, info = env.reset()
```

### 🗺️ Custom maps
Maps can be described by strings. We can either load them directly from a string or from a file.

```python
from generals.env import generals_v0
from generals.config import GameConfig

game_config = GameConfig(
    agent_names=['Human.exe','Agent007']  # Names of the agents that will be called to play the game
)
map = """
.3.#
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
- digits `0-9` represent cost of cities calculated as `(40 + digit)`

## 🔬 Replay analysis
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

generals.utils.run_replay("replay_001")
```
### 🕹️ Replay Controls
- `q` — quit/close the replay
- `←/→` — increase/decrease the replay speed
- `h/l` — to control replay frames
- `spacebar` — to pause
- `Mouse` click on the player's row — toggle the FOV (Field Of View) of the given player
## POMDP - 🔭 Observations, ⚡ Actions and 🎁 Rewards
### 🔭 Observation
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

### ℹ️ Information
The environment also returns information dictionary for each agent, but it is the same for everyone.
|Key|Type|Description|
|---|---|---|
|`army`|Int|Total number of units that the agent controls|
|`land`|Int|Total number of cells that the agent controls|
|`is_winner`|Bool|Boolean indicator saying whether agent won|

#### Example:
```python
print(info['red_agent']['army'])
```
   
### ⚡ Action
Action is an `np.array([i,j,k])` indicating that you want to move units from cell `[i,j]` in a direction `k`.

### 🎁 Reward
It is possible to implement custom reward function. The default is `1` for winner and `-1` for loser, otherwise `0`.
```python
def custom_reward_fn(observation, info):
    # Give agent a reward based on the number of cells they own
    return {
        agent: info[agent]["land"]
        for agent in observation.keys()
    }

env = generals_v0(reward_fn=custom_reward_fn)
observations, info = env.reset()
```
## 🔨 Coming soon:
- Extend action space to sending half of units to another square
- Examples and baselines using RL
- Add human control to play against
- New analysis tools
  
Requests for useful features and additions are welcome 🤗.
