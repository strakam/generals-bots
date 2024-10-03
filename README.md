<div align="center">

![Gameplay GIF](https://raw.githubusercontent.com/strakam/Generals-RL/master/generals/assets/gifs/wider_gameplay.gif)

## **Generals.io RL**

[![CodeQL](https://github.com/strakam/Generals-RL/actions/workflows/codeql.yml/badge.svg)](https://github.com/strakam/Generals-RL/actions/workflows/codeql.yml)
[![CI](https://github.com/strakam/Generals-RL/actions/workflows/tests.yml/badge.svg)](https://github.com/strakam/Generals-RL/actions/workflows/tests.yml)




[Installation](#-installation) • [Getting Started](#-getting-started) • [Customization](#-custom-maps) • [Environment](#-environment)
</div>

Generals-RL is a real-time strategy environment where players compete to conquer their opponents' generals on a 2D grid.
While the goal is simple — capture the enemy general — the gameplay involves a lot of depth.
Players need to employ strategic planning, deception, and manage both micro and macro mechanics throughout the game.
The combination of these elements makes the game highly engaging and complex.

This repository aims to make bot development more accessible, especially for Machine Learning based agents.

Highlights:
* 🚀 Fast & Lightweight simulator powered by `numpy` (thousands of steps per second)
* 🤝 Compatibility with Reinforcement-Learning API standards 🤸[Gymnasium](https://gymnasium.farama.org/) and 🦁[PettingZoo](https://pettingzoo.farama.org/)
* 🔧 Easy customization of environments
* 🔬 Analysis tools such as replays

> [!NOTE]
> This repository is based on the [generals.io](https://generals.io) game.
> Check it out, its a lot of fun !

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

## Usage example (🤸 Gymnasium)

```python
from generals import gym_generals
from generals.agents import RandomAgent, ExpanderAgent

# Initialize agents
agent = RandomAgent()
npc = ExpanderAgent()

# Create environment -- render modes: {None, "human"}
env = gym_generals(agent=agent, npc=npc, render_mode="human")
observation, info = env.reset()

done = False

while not done:
    action = agent.play(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render(fps=6)
```
You can also check an example for 🦁[PettingZoo](./examples/pettingzoo_example.py) or more extensive
example [here](./examples/complete_example.py).

## 🚀 Getting Started
Creating your first agent is very simple. 
- Start by subclassing an `Agent` class just like [`RandomAgent`](./generals/agents/random_agent.py) or [`ExpanderAgent`](./generals/agents/expander_agent.py).
- Every agent must have a name as it is his ID by which he is called for actions.
- Every agent must implement `play(observation)` function that takes in `observation` and returns an [action](#-action).
- You can start by copying the [Usage Example](#usage-example--gymnasium) and replacing `agent` with your implementation.
- When creating an environment, you can choose out of two `render_modes`:
     - `None` that omits rendering and is suitable for training,
     - `"human"` where you can see the game roll out.

> [!TIP]
> Check out `Makefile` and run some examples to get a feel for the game 🤗.

## 🎨 Custom grids
Grids are generated via `GridFactory`. You can instantiate the class with desired grid properties, and it will generate
grid with these properties for each run.
```python
from generals import pz_generals
from generals import GridFactory

grid_factory = GridFactory(
    grid_dims=(10, 10),                    # Dimensions of the grid (height, width)
    mountain_density=0.2,                  # Probability of a mountain in a cell
    city_density=0.05,                     # Probability of a city in a cell
    general_positions=[(0,3),(5,7)],       # Positions of generals (i, j)
)

# Create environment
env = pz_generals(grid_factory=grid_factory, ...)
```
You can also specify grids manually, as a string via `options` dict:
```python
from generals import pz_generals
from generals import GridFactory

grid_factory = GridFactory()
env = pz_generals(grid_factory=grid_factory, ...)

grid = """
.3.#
#..A
#..#
.#.B
"""

options = {"grid": grid}

# Pass the new grid to the environment (for the next game)
env.reset(options=options)
```
Grids are encoded using these symbols:
- `.` for cells where you can move your army
- `#` for mountains (terrain that can not be passed)
- `A,B` are positions of generals
- digits `0-9` represent cities with the cost calculated as `(40 + digit)`

## 🔬 Replays
We can store replays and then analyze them. `Replay` class handles replay related functionality.
### Storing a replay
```python
from generals import pz_generals

options = {"replay": "my_replay"}
env = pz_generals(...)
env.reset(options=options) # The next game will be encoded in my_replay.pkl
```

### Loading a replay

```python
from generals import Replay

# Initialize Replay instance
replay = Replay.load("my_replay")
replay.play()
```
### 🕹️ Replay controls
- `q` — quit/close the replay
- `r` — restart replay from the beginning
- `←/→` — increase/decrease the replay speed
- `h/l` — move backward/forward by one frame in the replay
- `spacebar` — toggle play/pause
- `mouse` click on the player's row — toggle the FoV (Field of View) of the given player

> [!WARNING]
> We are using the [pickle](https://docs.python.org/3/library/pickle.html) module which is not safe!
> Only open replays you trust.

## 🌍 Environment
### 🔭 Observation
An observation for one agent is a dictionary `{"observation": observation, "action_mask": action_mask}`.

The `observation` is a `Dict`. Values are either `numpy` matrices with shape `(N,M)`, or simple `int` constants:
| Key                  | Shape     | Description                                                                  |
| ---                  | ---       | ---                                                                          |
| `army`               | `(N,M)`   | Number of units in a cell regardless of owner                                |
| `general`            | `(N,M)`   | Mask of cells that are visible to the agent                                  |
| `city`               | `(N,M)`   | Mask saying whether a city is in a cell                                      |
| `visible_cells`      | `(N,M)`   | Mask indicating cells that are visible to the agent                          |
| `owned_cells`        | `(N,M)`   | Mask indicating cells controlled by the agent                                |
| `opponent_cells`     | `(N,M)`   | Mask indicating cells owned by the opponent                                  |
| `neutral_cells`      | `(N,M)`   | Mask indicating cells that are not owned by agents                           |
| `structure`          | `(N,M)`   | Mask indicating whether cells contain cities or mountains, even out of FoV   |
| `owned_land_count`   |     —     | Number of cells an agent owns                                                |
| `owned_army_count`   |     —     | Total number of units of an agent over all cells                             |
| `opponent_land_count`|     —     | Number of cells owned by the opponent                                        |
| `opponent_army_count`|     —     | Int representing total number of units owned by the opponent                 |
| `is_winner`          |     —     | Whether agent won                                                            |
| `timestep`           |     —     | Timestep                                                                     |

`action_mask` is a mask with shape `(N,M,4)` where value `[i,j,d]` says whether you can move from cell `[i,j]` in a direction `d`.
   
### ⚡ Action
Action is a `tuple(pass, cell, direction, split)`, where:
- `pass` indicates whether you want to `1 (pass)` or `0 (play)`.
- `cell` is an `np.array([i,j])` where `i,j` are indices of the cell you want to move from
- `direction` indicates whether you want to move `0 (up)`, `1 (down)`, `2 (left)`, or `3 (right)`
- `split` indicates whether you want to `1 (split)` units (send half of them) or `0 (no split)`, which sends all possible units to the next cell.

> [!TIP]
> You can see how actions and observations look like by printing a sample form the environment:
> ```python
> print(env.observation_space.sample())
> print(env.action_space.sample())
> ```

### 🎁 Reward
It is possible to implement custom reward function. The default is `1` for winner and `-1` for loser, otherwise `0`.
```python
def custom_reward_fn(observation, action, done, info):
    # Give agent a reward based on the number of cells they own
    return observation["observation"]["owned_land_count"]

env = pz_generals(reward_fn=custom_reward_fn)
observations, info = env.reset()
```
