import copy
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from generals.gui import GUI
from generals.gui.event_handler import ReplayCommand
from generals.gui.properties import GuiMode
from generals.rewards.reward_fn import RewardFn
from generals.rewards.win_lose_reward_fn import WinLoseRewardFn

from .action import Action
from .channels import Channels
from .config import DIRECTIONS
from .grid import Grid, GridFactory
from .observation import Observation


class Environment:
    """
    This class represents the Environment we're trying to optimize and accordingly manages the bulk
    of the logic for the game -- generals.io. Like any reinforcement-learning Environment, it accepts actions
    at each timestep and gives rise to new observations and rewards based on those actions.

    When implementing new Environment classes for existing RL frameworks (e.g. Gymnasium, PettingZoo,
    RLLib), this class should manage all game-related logic. And that new class should only modify
    the outputs of this class to cater to each libraries specific expectations. For examples of how
    this can be done, look to the currently available env implementations in generals/envs.
    """

    # Generals games are extremely unlikely to need values beyond these. However,
    # they may still be tweaked if desired. The environment/simulator will
    # crash if any of these values are exceeded.
    max_army_size = 100_000
    max_timestep = 100_000
    max_land_owned = 250 * 250

    # Every {increment_rate} turns, each land-tile that's owned
    # generates an army.
    increment_rate = 50

    # Default fps of the GUI. Can be modified via speed_multiplier.
    default_render_fps = 6

    def __init__(
        self,
        agent_ids: list[str],
        grid_factory: GridFactory = None,
        truncation: int = None,
        reward_fn: RewardFn = None,
        to_render: bool = False,
        speed_multiplier: float = 1.0,
        save_replays: bool = False,
    ):
        self.agent_ids = agent_ids
        self.grid_factory = grid_factory if grid_factory is not None else GridFactory()
        self.truncation = truncation
        self.reward_fn = reward_fn if reward_fn is not None else WinLoseRewardFn()
        self.to_render = to_render
        self.speed_multiplier = speed_multiplier
        self.save_replays = save_replays

        self.episode_num = 0
        self._reset()

    def render(self):
        if self.to_render:
            fps = int(self.default_render_fps * self.speed_multiplier)
            _ = self.gui.tick(self.get_infos(), self.num_turns, fps)

    def close(self):
        if self.to_render:
            self.gui.close()

    def reset_from_gymnasium(self, rng: np.random.Generator, options: dict[str, Any] = None):
        """Reset this environment in accordance with Gymnasium's env-resetting expectations."""

        if options is not None and "grid" in options:
            grid = Grid(options["grid"])
        else:
            # Provide the np.random.Generator instance created in Env.reset()
            # as opposed to creating a new one with the same seed.
            self.grid_factory.set_rng(rng=rng)
            grid = self.grid_factory.generate()

        self._reset(grid, options)

        observation = self.agent_observation(self.agent_ids[0])
        info = {}  # type: ignore
        return observation, info

    def reset_from_petting_zoo(self, seed: int = None, options: dict[str, Any] = None):
        """Reset this environment in accordance with PettingZoo's env-resetting expectations."""

        if options is not None and "grid" in options:
            grid = Grid(options["grid"])
        else:
            # The pettingzoo.Parallel_Env's reset() notably differs
            # from gymnasium.Env's reset() in that it does not create
            # a random generator which should be re-used.
            self.grid_factory.set_rng(rng=np.random.default_rng(seed))
            grid = self.grid_factory.generate()

        self._reset(grid, options)

        observations = {agent_id: self.agent_observation(agent_id) for agent_id in self.agent_ids}
        infos = {agent_id: {} for agent_id in self.agent_ids}  # type: ignore

        return observations, infos

    def _reset(self, grid: Grid = None, options: dict[str, Any] = None):
        """_reset contains instructions common for resetting all types of envs."""

        # Observations for each agent at the prior time-step.
        self.prior_observations: dict[str, Observation] = None

        # The priority-ordering of each agent. This determines which agents' action is processed first.
        self.agents_in_order_of_prio = self.agent_ids[:]

        # The number of turns and the time displayed in game differ. In generals.io there are two turns
        # each agent may take per in-game unit of time.
        self.num_turns = 0

        # Reset the grid & channels, i.e. the game-board.
        if grid is None:
            grid = self.grid_factory.generate()
        self.channels = Channels(grid.grid, self.agent_ids)
        self.grid_dims = (grid.grid.shape[0], grid.grid.shape[1])

        general_positions = grid.get_generals_positions()
        self.general_positions = {self.agent_ids[idx]: general_positions[idx] for idx in range(0, 2)}

        # Reset the GUI for the upcoming game.
        if self.to_render:
            self.gui = GUI(self.channels, self.agent_ids, self.grid_dims, GuiMode.TRAIN)

        # Prepare a new replay to save the upcoming game.
        if self.save_replays:
            self.replay = Replay(self.episode_num, grid, self.agent_ids)
            self.replay.add_state(self.channels)

        self.episode_num += 1

    def step(
        self, actions: dict[str, Action]
    ) -> tuple[dict[str, Observation], dict[str, Any], bool, bool, dict[str, Any]]:
        """
        Perform one step of the game
        """
        done_before_actions = self.is_done()
        for agent in self.agents_in_order_of_prio:
            pass_turn, si, sj, direction, split_army = actions[agent]

            # Skip if agent wants to pass the turn
            if pass_turn == 1:
                continue
            if split_army == 1:  # Agent wants to split the army
                army_to_move = self.channels.armies[si, sj] // 2
            else:  # Leave just one army in the source cell
                army_to_move = self.channels.armies[si, sj] - 1
            if army_to_move < 1:  # Skip if army size to move is less than 1
                continue

            # Cap the amount of army to move (previous moves may have lowered available army)
            army_to_move = min(army_to_move, self.channels.armies[si, sj] - 1)
            army_to_stay = self.channels.armies[si, sj] - army_to_move

            # Check if the current agent still owns the source cell and has more than 1 army
            if self.channels.ownership[agent][si, sj] == 0 or army_to_move < 1:
                continue

            di, dj = (
                si + DIRECTIONS[direction].value[0],
                sj + DIRECTIONS[direction].value[1],
            )  # destination indices

            # Skip if the destination cell is not passable or out of bounds
            if di < 0 or di >= self.grid_dims[0] or dj < 0 or dj >= self.grid_dims[1]:
                continue
            if self.channels.passable[di, dj] == 0:
                continue

            # Figure out the target square owner and army size
            target_square_army = self.channels.armies[di, dj]
            target_square_owner_idx = np.argmax(
                [self.channels.ownership[agent][di, dj] for agent in ["neutral"] + self.agent_ids]
            )
            target_square_owner = (["neutral"] + self.agent_ids)[target_square_owner_idx]
            if target_square_owner == agent:
                self.channels.armies[di, dj] += army_to_move
                self.channels.armies[si, sj] = army_to_stay
            else:
                # Calculate resulting army, winner and update channels
                remaining_army = np.abs(target_square_army - army_to_move)
                square_winner = agent if target_square_army < army_to_move else target_square_owner
                self.channels.armies[di, dj] = remaining_army
                self.channels.armies[si, sj] = army_to_stay
                self.channels.ownership[square_winner][di, dj] = True
                if square_winner != target_square_owner:
                    self.channels.ownership[target_square_owner][di, dj] = False

        # Swap agent order (because priority is alternating)
        self.agents_in_order_of_prio = self.agents_in_order_of_prio[::-1]

        if not done_before_actions:
            self.num_turns += 1

        if self.is_done():
            # give all cells of loser to winner
            winner = self.agent_ids[0] if self.agent_won(self.agent_ids[0]) else self.agent_ids[1]
            loser = self.agent_ids[1] if winner == self.agent_ids[0] else self.agent_ids[0]
            self.channels.ownership[winner] += self.channels.ownership[loser]
            self.channels.ownership[loser] = np.full(self.grid_dims, False)
        else:
            self._global_game_update()

        observations = {agent: self.agent_observation(agent) for agent in self.agent_ids}
        infos = self.get_infos()

        if self.prior_observations is None:
            # Cannot compute rewards without prior-observations. This should only happen
            # on the first time-step.
            rewards = {agent: 0.0 for agent in self.agent_ids}
        else:
            rewards = {
                agent: self.reward_fn(
                    prior_obs=self.prior_observations[agent],
                    # Technically actions are the prior-actions, since they are what will give
                    # rise to the current-observations.
                    prior_action=actions[agent],
                    obs=observations[agent],
                )
                for agent in self.agent_ids
            }

        terminated = self.is_done()
        truncated = False
        if self.truncation is not None:
            truncated = self.num_turns >= self.truncation

        if self.save_replays:
            self.replay.add_state(self.channels)

        if (terminated or truncated) and self.save_replays:
            self.replay.store()

        self.prior_observations = observations

        return observations, rewards, terminated, truncated, infos

    def _global_game_update(self) -> None:
        """
        Update game state globally.
        """

        owners = self.agent_ids

        # every `increment_rate` steps, increase army size in each cell
        if self.num_turns % self.increment_rate == 0:
            for owner in owners:
                self.channels.armies += self.channels.ownership[owner]

        # Increment armies on general and city cells, but only if they are owned by player
        if self.num_turns % 2 == 0 and self.num_turns > 0:
            update_mask = self.channels.generals + self.channels.cities
            for owner in owners:
                self.channels.armies += update_mask * self.channels.ownership[owner]

    def is_done(self) -> bool:
        """
        Returns True if the game is over, False otherwise.
        """
        return any(self.agent_won(agent) for agent in self.agent_ids)

    def get_infos(self) -> dict[str, dict[str, Any]]:
        """
        Returns a dictionary of player statistics.
        Keys and values are as follows:
        - army: total army size
        - land: total land size
        - is_done: True if the game is over, False otherwise
        - is_winner: True if the player won, False otherwise
        """
        players_stats = {}
        for agent in self.agent_ids:
            army_size = np.sum(self.channels.armies * self.channels.ownership[agent]).astype(int)
            land_size = np.sum(self.channels.ownership[agent]).astype(int)
            players_stats[agent] = {
                "army": army_size,
                "land": land_size,
                "is_done": self.is_done(),
                "is_winner": self.agent_won(agent),
            }
        return players_stats

    def agent_observation(self, agent: str) -> Observation:
        """
        Returns an observation for a given agent.
        """
        scores = {}
        for _agent in self.agent_ids:
            army_size = np.sum(self.channels.armies * self.channels.ownership[_agent]).astype(int)
            land_size = np.sum(self.channels.ownership[_agent]).astype(int)
            scores[_agent] = {
                "army": army_size,
                "land": land_size,
            }

        visible = self.channels.get_visibility(agent)
        invisible = 1 - visible

        opponent = self.agent_ids[0] if agent == self.agent_ids[1] else self.agent_ids[1]

        armies = self.channels.armies.astype(int) * visible
        mountains = self.channels.mountains * visible
        generals = self.channels.generals * visible
        cities = self.channels.cities * visible
        neutral_cells = self.channels.ownership_neutral * visible
        owned_cells = self.channels.ownership[agent] * visible
        opponent_cells = self.channels.ownership[opponent] * visible
        structures_in_fog = invisible * (self.channels.mountains + self.channels.cities)
        fog_cells = invisible - structures_in_fog
        owned_land_count = scores[agent]["land"]
        owned_army_count = scores[agent]["army"]
        opponent_land_count = scores[opponent]["land"]
        opponent_army_count = scores[opponent]["army"]
        timestep = self.num_turns
        priority = 1 if agent == self.agents_in_order_of_prio[0] else 0

        return Observation(
            armies=armies,
            generals=generals,
            cities=cities,
            mountains=mountains,
            neutral_cells=neutral_cells,
            owned_cells=owned_cells,
            opponent_cells=opponent_cells,
            fog_cells=fog_cells,
            structures_in_fog=structures_in_fog,
            owned_land_count=owned_land_count,
            owned_army_count=owned_army_count,
            opponent_land_count=opponent_land_count,
            opponent_army_count=opponent_army_count,
            timestep=timestep,
            priority=priority,
        )

    def agent_won(self, agent: str) -> bool:
        """
        Returns True if the agent won the game, False otherwise.
        """
        return all(
            self.channels.ownership[agent][general[0], general[1]] == 1 for general in self.general_positions.values()
        )


class Replay:
    replays_dir = Path.cwd() / "replays"

    def __init__(self, episode_num: int, grid: Grid, agent_ids: list[str]):
        # Create the replays/ directory if necessary.
        self.replays_dir.mkdir(parents=True, exist_ok=True)

        self.replay_filename = self.replays_dir / f"replay_{episode_num}.pkl"
        self.grid = grid
        self.agent_ids = agent_ids

        self.game_states: list[Channels] = []

    def add_state(self, state: Channels):
        copied_state = copy.deepcopy(state)
        self.game_states.append(copied_state)

    def store(self):
        with open(self.replay_filename, "wb") as f:
            pickle.dump(self, f)
        print(f"Replay saved to {self.replay_filename}.")

    @classmethod
    def load(cls, path):
        path = path if path.endswith(".pkl") else path + ".pkl"
        with open(path, "rb") as f:
            return pickle.load(f)

    def play(self):
        agents = [agent for agent in self.agent_ids]
        game = Environment(self.grid, agents)
        gui = GUI(game.channels, agents, game.grid_dims, gui_mode=GuiMode.REPLAY)
        gui_properties = gui.properties

        game_step, last_input_time, last_move_time = 0, 0, 0
        while True:
            _t = time.time()

            # Check inputs
            if _t - last_input_time > 0.008:  # check for input every 8ms
                command = gui.tick()
                last_input_time = _t
            else:
                command = ReplayCommand()

            if command.restart:
                game_step = 0

            # If we control replay, change game state
            game_step = max(0, min(len(self.game_states) - 1, game_step + command.frame_change))
            if gui_properties.paused and game_step != game.num_turns:
                game.channels = copy.deepcopy(self.game_states[game_step])
                game.num_turns = game_step
                last_move_time = _t
            # If we are not paused, play the game
            elif _t - last_move_time > (1 / gui_properties.game_speed) * 0.512 and not gui_properties.paused:
                if game.is_done():
                    gui_properties.paused = True
                game_step = min(len(self.game_states) - 1, game_step + 1)
                game.channels = copy.deepcopy(self.game_states[game_step])
                game.num_turns = game_step
                last_move_time = _t
            gui_properties.clock.tick(60)
