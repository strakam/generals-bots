import generals.utils
from generals.env import GameConfig


testik = GameConfig(replay_file='test')
agents = [generals.utils.Player("red"), generals.utils.Player("blue")]

# generals.utils.run_from_replay('test', agents)
generals.utils.run(testik, agents)
