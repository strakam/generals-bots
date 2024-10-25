<div align="center">

![Gameplay GIF](https://raw.githubusercontent.com/strakam/generals-bots/master/generals/assets/gifs/wider_gameplay.gif)

## **Generals.io Bots**

[![CodeQL](https://github.com/strakam/generals-bots/actions/workflows/codeql.yml/badge.svg)](https://github.com/strakam/generals-bots/actions/workflows/codeql.yml)
[![CI](https://github.com/strakam/generals-bots/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/strakam/generals-bots/actions/workflows/pre-commit.yml)




[Installation](#-installation) • [Getting Started](#-getting-started) • [Customization](#-custom-grids) • [Environment](#-environment) • [Wiki](https://github.com/strakam/generals-bots/wiki)
</div>

Generals-bots is a fast-paced strategy environment where players compete to conquer their opponents' generals on a 2D grid.
While the goal is simple — capture the enemy general — the gameplay combines strategic depth with fast-paced action,
challenging players to balance micro and macro-level decision-making.
The combination of these elements makes the game highly engaging and complex.

Highlights:
* ⚡ **blazing-fast simulator**: run thousands of steps per second with `numpy`-powered efficiency
* 🤝 **seamless integration**: fully compatible with RL standards 🤸[Gymnasium](https://gymnasium.farama.org/) and 🦁[PettingZoo](https://pettingzoo.farama.org/)
* 🔧 **extensive customization**: easily tailor environments to your specific needs
* 🚀 **effortless deployment**: launch your agents to [generals.io](https://generals.io)
* 🔬 **analysis tools**: leverage features like replays for deeper insights

> [!Note]
> This repository is based on the [generals.io](https://generals.io) game (check it out, it's a lot of fun!).
> The one and only goal of this project is to provide a bot development platform, especially for Machine Learning based agents.

## 📦 Installation
You can install the latest stable version via `pip` for reliable performance
```bash
pip install generals-bots
```
or clone the repo for the most up-to-date features
```bash
git clone https://github.com/strakam/generals-bots
cd generals-bots
make install
```
> [!Note]
> Under the hood, `make install` installs [poetry](https://python-poetry.org/) and the package using `poetry`.

## 🌱 Getting Started
Creating an agent is very simple. Start by subclassing an `Agent` class just like
[`RandomAgent`](./generals/agents/random_agent.py) or [`ExpanderAgent`](./generals/agents/expander_agent.py).
You can specify your agent `id` (name) and `color` and the only thing remaining is to implement the `act` function,
that has the signature explained in sections down below.


### Usage Example (🤸 Gymnasium)
The example loop for running the game looks like this
```python
import gymnasium as gym
from generals.agents import RandomAgent, ExpanderAgent # import your agent

# Initialize agents
agent = RandomAgent()
npc = ExpanderAgent()

# Create environment
env = gym.make("gym-generals-v0", agent=agent, npc=npc, render_mode="human")

observation, info = env.reset()
terminated = truncated = False
while not (terminated or truncated):
    action = agent.act(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
```

> [!TIP]
> Check out [Wiki](https://github.com/strakam/generals-bots/wiki) for more commented examples to get a better idea on how to start 🤗.

## 🎨 Custom Grids
Grids on which the game is played on are generated via `GridFactory`. You can instantiate the class with desired grid properties, and it will generate
grid with these properties for each run.
```python
import gymnasium as gym
from generals import GridFactory

grid_factory = GridFactory(
    grid_dims=(10, 10),                    # Dimensions of the grid (height, width)
    mountain_density=0.2,                  # Probability of a mountain in a cell
    city_density=0.05,                     # Probability of a city in a cell
    general_positions=[(0,3),(5,7)],       # Positions of generals (i, j)
)

# Create environment
env = gym.make(
    "gym-generals-v0",
    grid_factory=grid_factory,
    ...
)
```
You can also specify grids manually, as a string via `options` dict:
```python
import gymnasium as gym

env = gym.make("gym-generals-v0", ...)
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
Grids are created using a string format where:
- `.` represents passable terrain
- `#` indicates impassable mountains
- `A, B` mark the positions of generals
- digits `0-9` represent cities, where the number specifies amount of neutral army in the city,
  which is calculated as `40 + digit`

## 🔬 Interactive Replays
We can store replays and then analyze them in an interactive fashion. `Replay` class handles replay related functionality.
### Storing a replay
```python
import gymnasium as gym

env = gym.make("gym-generals-v0", ...)

options = {"replay": "my_replay"}
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
You can control your replays to your liking! Currently, we support these controls:
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
| -------------------- | --------- | ---------------------------------------------------------------------------- |
| `armies`             | `(N,M)`   | Number of units in a visible cell regardless of the owner                    |
| `generals`           | `(N,M)`   | Mask indicating visible cells containing a general                           |
| `cities`             | `(N,M)`   | Mask indicating visible cells containing a city                              |
| `mountains`          | `(N,M)`   | Mask indicating visible cells containing mountains                           |
| `neutral_cells`      | `(N,M)`   | Mask indicating visible cells that are not owned by any agent                |
| `owned_cells`        | `(N,M)`   | Mask indicating visible cells owned by the agent                             |
| `opponent_cells`     | `(N,M)`   | Mask indicating visible cells owned by the opponent                          |
| `fog_cells`          | `(N,M)`   | Mask indicating fog cells that are not mountains or cities                   |
| `structures_in_fog`  | `(N,M)`   | Mask showing cells containing either cities or mountains in fog              |
| `owned_land_count`   |     —     | Number of cells the agent owns                                               |
| `owned_army_count`   |     —     | Total number of units owned by the agent                                     |
| `opponent_land_count`|     —     | Number of cells owned by the opponent                                        |
| `opponent_army_count`|     —     | Total number of units owned by the opponent                                  |
| `timestep`           |     —     | Current timestep of the game                                                 |

The `action_mask` is a 3D array with shape `(N, M, 4)`, where each element corresponds to whether a move is valid from cell
`[i, j]` in one of four directions: `0 (up)`, `1 (down)`, `2 (left)`, or `3 (right)`.

### ⚡ Action
Actions are in a `dict` format with the following `key: value` format:
- `pass` indicates whether you want to `1 (pass)` or `0 (play)`.
- `cell` value is an `np.array([i,j])` where `i,j` are indices of the cell you want to move from
- `direction` indicates whether you want to move `0 (up)`, `1 (down)`, `2 (left)`, or `3 (right)`
- `split` indicates whether you want to `1 (split)` units and send only half, or `0 (no split)` where you send all units to the next cell

> [!TIP]
> You can see how actions and observations look like by printing a sample form the environment:
> ```python
> print(env.observation_space.sample())
> print(env.action_space.sample())
> ```

### 🎁 Reward
It is possible to implement custom reward function. The default reward is awarded only at the end of a game
and gives `1` for winner and `-1` for loser, otherwise `0`.
```python
def custom_reward_fn(observation, action, done, info):
    # Give agent a reward based on the number of cells they own
    return observation["observation"]["owned_land_count"]

env = gym.make(..., reward_fn=custom_reward_fn)
observations, info = env.reset()
```

## 🚀 Deployment to Live Servers
Complementary to local development, it is possible to run agents online against other agents and players.
We use `socketio` for communication, and you can either use our `autopilot` to run agent in a specified lobby indefinitely,
or create your own connection workflow. Our implementations expect that your agent inherits from the `Agent` class, and has
implemented the required methods.
```python
from generals.remote import autopilot
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--user_id", type=str, default=...) # Register yourself at generals.io and use this id
parser.add_argument("--lobby_id", type=str, default="psyo") # The last part of the lobby url
parser.add_argument("--agent_id", type=str, default="Expander") # agent_id should be "registered" in AgentFactory

if __name__ == "__main__":
    args = parser.parse_args()
    autopilot(args.agent_id, args.user_id, args.lobby_id)
```
This script will run your `ExpanderAgent` in lobby `psyo`.
## 🙌 Contributing
You can contribute to this project in multiple ways:
- 🤖 If you implement ANY non-trivial agent, send it to us! We will publish it, so others can play against it.
- 💡 If you have an idea on how to improve the game, submit an issue or create a PR, we are happy to improve!
  We also have some ideas (see [issues](https://github.com/strakam/generals-bots/issues)), so you can see what we plan to work on.

> [!Tip]
> Check out [wiki](https://github.com/strakam/generals-bots/wiki) to learn in more detail on how to contribute.
