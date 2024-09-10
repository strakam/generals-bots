<div align="center">

[<img src="https://github.com/strakam/Generals-Zoo/blob/master/generals/images/test.png?raw=true" alt="Generals-Zoo logo" width="500"/>](https://github.com/strakam/Generals-Zoo)

**Machine Learning Friendly Implementation of the Generals.io Game**
 
</div>

The [generals.io](https://generals.io/) is a real-time strategy game based on conquest and tactical maneuvering where players
are placed on a 2D grid with only one goal - to capture their opponents' general. Behind this simple goal is a lot of strategic planning, deception, micro and macro mechanics and more. The game has many interesting properties from the view of bot development.
It is simple to control, partially observable, real-time, has long action sequences and the action space is large.

Although [generals.io](https://generals.io/) enjoys an active player base that organizes tournaments, its bot landscape is significantly smaller. Currently there is only one, "rule-based" agent capable of consistently beating humans. But one question
is still not answered. Can Machine Learning agents powered by recent advances in (Deep) Reinforcement Learning beat curretn State-of-the-Art? One barrier to answering this question is that it is very difficult to start implementing an agent with the official botting API, especially for Machine Learning agents, who want to be ML framework friendly. This project aims to remove this barrier by implementing the `numpy` powered environment, fully compatible with the [PettingZoo](https://pettingzoo.farama.org/) API standard, which makes agent development as easy as implementing one function. 


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
