#!/usr/bin/env python3
import argparse
import collections
from typing import Self
from generals import GridFactory

import gymnasium as gym
import numpy as np
import torch

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument(
    "--recodex", default=False, action="store_true", help="Running in ReCodEx"
)
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument(
    "--threads", default=8, type=int, help="Maximum number of threads to use."
)
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--epsilon", default=0.2, type=float, help="Exploration factor.")
parser.add_argument(
    "--epsilon_final", default=0.1, type=float, help="Final exploration factor."
)
parser.add_argument(
    "--epsilon_final_at", default=8000, type=int, help="Training episodes."
)
parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
parser.add_argument(
    "--hidden_layer_size", default=100, type=int, help="Size of hidden layer."
)
parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate.")
parser.add_argument(
    "--target_update_freq", default=200, type=int, help="Target update frequency."
)


class Network:
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Create a suitable model and store it as `self._model`.
        # Make U-Net like architecture
        self._model = torch.nn.Sequential(
            torch.nn.Conv2d(14, 32, kernel_size=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=2),
            torch.nn.ReLU(),  # now start upsampling
            torch.nn.ConvTranspose2d(128, 64, kernel_size=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 4, kernel_size=2),
        )

        # TODO: Define an optimizer (most likely from `torch.optim`).
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=args.learning_rate
        )

        # TODO: Define the loss (most likely some `torch.nn.*Loss`).
        self._loss = torch.nn.MSELoss()

        # PyTorch uses uniform initializer $U[-1/sqrt n, 1/sqrt n]$ for both weights and biases.
        # Keras uses Glorot (also known as Xavier) uniform for weights and zeros for biases.
        # In some experiments, the Keras initialization works slightly better for RL,
        # so we use it instead of the PyTorch initialization; but feel free to experiment.
        self._model.apply(wrappers.torch_init_with_xavier_and_zeros)

    # Define a training method. Generally you have two possibilities
    # - pass new q_values of all actions for a given state; all but one are the same as before
    # - pass only one new q_value for a given state, and include the index of the action to which
    #   the new q_value belongs
    # The code below implements the first option, but you can change it if you want.
    #
    # The `wrappers.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @wrappers.typed_torch_function(device, torch.float32, torch.float32)
    def train(self, states: torch.Tensor, q_values: torch.Tensor) -> None:
        self._model.train()
        self._optimizer.zero_grad()
        predictions = self._model(states)
        predictions = predictions.view(predictions.shape[0], -1)
        loss = self._loss(predictions, q_values)
        loss.backward()
        with torch.no_grad():
            self._optimizer.step()

    @wrappers.typed_torch_function(device, torch.float32)
    def predict(self, states: torch.Tensor) -> np.ndarray:
        self._model.eval()
        with torch.no_grad():
            grid_prediction = self._model(states)
            return grid_prediction.view(grid_prediction.shape[0], -1)

    # If you want to use target network, the following method copies weights from
    # a given Network to the current one.
    def copy_weights_from(self, other: Self) -> None:
        self._model.load_state_dict(other._model.state_dict())


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set random seeds and the number of threads
    np.random.seed(args.seed)

    if args.seed is not None:
        torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    # Construct the network
    network = Network(env, args)
    target_network = Network(env, args)
    target_network.copy_weights_from(network)

    # Replay memory; the `max_length` parameter can be passed to limit its size.
    replay_buffer = wrappers.ReplayBuffer(max_length=1_000_000)
    Transition = collections.namedtuple(
        "Transition", ["state", "action", "reward", "done", "next_state"]
    )

    epsilon = args.epsilon
    training = True
    ep = 0
    rolling_score = np.array([])
    while training:
        # Perform episode
        state, done = env.reset()[0], False
        ep_reward = 0
        ep += 1
        while not done:
            if ep % args.target_update_freq == 0:
                target_network.copy_weights_from(network)
            # TODO: Choose an action.
            q_values = network.predict(state[np.newaxis])
            action = (
                np.argmax(q_values)
                if np.random.rand() > epsilon
                else np.random.randint(64)
            )
            i, j, d = np.unravel_index(action, (4, 4, 4))

            next_state, reward, terminated, truncated, _ = env.step(
                (0, np.array([i, j]), d, 0)
            )
            done = terminated or truncated
            ep_reward += reward

            # Append state, action, reward, done and next_state to replay_buffer
            replay_buffer.append(Transition(state, action, reward, done, next_state))

            # TODO: If the `replay_buffer` is large enough, perform training using
            # a batch of `args.batch_size` uniformly randomly chosen transitions.
            #
            if len(replay_buffer) >= 1_000:
                batch = replay_buffer.sample(
                    args.batch_size, generator=np.random, replace=True
                )
                actions = np.array([t.action for t in batch])
                states = np.array([t.state for t in batch])
                dones = np.array([t.done for t in batch])
                rewards = np.array([t.reward for t in batch])
                next_states = np.array([t.next_state for t in batch])

                targets = network.predict(states)
                new_values = (
                    args.gamma
                    * np.max(target_network.predict(next_states), axis=-1)
                    * (1 - dones)
                    + rewards
                )
                targets[np.arange(len(actions)), actions] = new_values
                network.train(states, targets)

            state = next_state
        if ep % 10 == 0:
            print(f"Episode {ep}, reward {np.mean(rolling_score[-100:])}")

        rolling_score = np.append(rolling_score, ep_reward)
        if len(rolling_score) > 100:
            rolled = np.mean(rolling_score[-100:])
            if rolled >= 480:
                training = False

        # if args.epsilon_final_at:
        #     epsilon = np.interp(env.episode + 1, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # TODO: Choose (greedy) action
            q_values = network.predict(state[np.newaxis])[0]
            action = np.argmax(q_values)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    # save torch model


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    grid_factory = GridFactory(
        grid_dims=(4, 4),
        general_positions=[(0, 0), (3, 3)],
        seed=args.seed,
    )


    def reward_fn(observation, action, done, info):
        my_land = observation["observation"]["owned_land_count"]
        opponent_land = observation["observation"]["opponent_land_count"]
        return (my_land - opponent_land) / 16

    env = gym.make("gym-generals-v0", grid_factory=grid_factory, reward_fn=reward_fn)

    main(env, args)
