<div align="center">

[<img src="https://github.com/strakam/Generals-Zoo/blob/master/generals/images/test.png?raw=true" alt="Generals-Zoo logo" width="500"/>](https://github.com/strakam/Generals-Zoo)

**Machine Learning Friendly Implementation of the Generals.io Game**
 
</div>

[generals.io](https://generals.io/) is a real-time strategy game where players compete to conquer their opponents' generals on a 2D grid. While the goal is simple—capture the enemy general—the gameplay involves a lot of depth. Players need to employ strategic planning, deception, and manage both micro and macro mechanics throughout the game. The combination of these elements makes the game highly engaging and complex.

From a bot development perspective, generals.io has some fascinating characteristics. It’s easy to control but challenging due to its partially observable environment, real-time dynamics, long sequences of actions, and a large action space.

While the game has an active player base, with tournaments being held regularly, the bot landscape remains quite limited. Currently, there’s only one "rule-based" bot that can consistently beat human players. This raises an intriguing question: can Machine Learning (ML) agents, powered by recent advances in (Deep) Reinforcement Learning, outperform the current state-of-the-art bot?

One challenge to answering this question is that the official botting API makes it tough to implement ML-based agents, particularly for developers who want to integrate modern ML frameworks. To solve this, our project aims to make bot development more accessible by creating a numpy-powered environment that is fully compatible with the [PettingZoo](https://pettingzoo.farama.org/) API standard. This simplifies the development process, making it as easy as implementing a single function.


## Installation/Setup
TODO

## Running the game

Run the game (and record it):
```sh
make run
```

Replay the last game:
```sh
make t_replay
```

## Controls
- `q` — quit/close the game
- `←/→` — increase/decrease the game speed
- `h/l` - to control replay frames
- `spacebar` - to pause
- `Mouse` click on the player's row — toggle the FOV (Field Of View) of the given player
