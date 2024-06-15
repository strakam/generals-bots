from generals.env import generals_v0
import generals.config as game_config
from pettingzoo.test import parallel_api_test

if __name__ == "__main__":
    config = game_config.Config(
        grid_size=10,
        starting_positions=[[1, 1], [5, 5]],
    )
    env = generals_v0(config, render_mode="none")
    # test the environment with parallel_api_test
    import time
    start = time.time()
    parallel_api_test(env, num_cycles=100_000)
    print("Time taken: ", time.time() - start)
    env.close()
