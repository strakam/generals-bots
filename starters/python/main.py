"""
Python starter — game-loop scaffolding.

You shouldn't need to edit this file. Implement your strategy in `agent.py`.

This file does three things:
  1. Defines the `Observation` dataclass that holds one turn's parsed view.
  2. Parses the wire-format frame from stdin into an `Observation`.
  3. Drives the per-turn loop: read frame, call `Agent.act`, write action.

Protocol summary (see generals/protocol.py in the engine repo for the spec):
    Handshake (engine -> agent, once):
        <player_id> <H> <W>
    Per turn (engine -> agent):
        <turn> <my_land> <my_army> <opp_land> <opp_army>
        H lines of W ints   (type grid)
        H lines of W ints   (owner grid)
        H lines of W ints   (army grid)
    Per turn (agent -> engine):
        <pass> <row> <col> <dir> <split>
    Game end: the engine closes our stdin. We exit on EOF.
"""
import sys
from dataclasses import dataclass
from typing import List

from agent import Agent


# Minimal, from-scratch observation type. It only carries what the wire
# format gives you — no JAX, no numpy, no engine internals.
@dataclass
class Observation:
    H: int                       # board height
    W: int                       # board width
    turn: int                    # current turn number
    my_land: int                 # cells you own
    my_army: int                 # total armies on cells you own
    opp_land: int                # opponent's land count
    opp_army: int                # opponent's army total
    type_grid: List[List[int]]   # 0=fog 1=plain 2=mountain 3=city 4=general 5=structure-in-fog
    owner_grid: List[List[int]]  # 0=neutral/unknown 1=me 2=opp (perspective-relative)
    army_grid: List[List[int]]   # army count per cell, 0 if not visible


def _read_grid(stdin, H):
    # One row of W integers per line; H lines total.
    return [[int(x) for x in stdin.readline().split()] for _ in range(H)]


def _read_observation(stdin, H, W, scalars_line):
    # `scalars_line` is the already-consumed first line of the frame.
    turn, my_land, my_army, opp_land, opp_army = (int(x) for x in scalars_line.split())
    type_grid = _read_grid(stdin, H)
    owner_grid = _read_grid(stdin, H)
    army_grid = _read_grid(stdin, H)
    return Observation(
        H=H, W=W, turn=turn,
        my_land=my_land, my_army=my_army,
        opp_land=opp_land, opp_army=opp_army,
        type_grid=type_grid,
        owner_grid=owner_grid,
        army_grid=army_grid,
    )


def main():
    stdin = sys.stdin
    stdout = sys.stdout

    # Handshake arrives once, before any frames. We learn our player id and
    # the board dimensions (both stay constant for the whole game).
    handshake = stdin.readline()
    if not handshake:
        return
    player_id, H, W = (int(x) for x in handshake.split())

    agent = Agent(player_id=player_id, H=H, W=W)

    while True:
        # readline returns "" on EOF (i.e., when the runner closes our stdin
        # because the game is over). Anything else is the scalars line of a
        # new observation frame.
        first = stdin.readline()
        if not first:
            return

        obs = _read_observation(stdin, H, W, first)
        p, r, c, d, s = agent.act(obs)

        # Flush is mandatory: pipes are fully buffered by default, so without
        # it the runner never sees our action and both sides deadlock.
        stdout.write(f"{p} {r} {c} {d} {s}\n")
        stdout.flush()


if __name__ == "__main__":
    main()
