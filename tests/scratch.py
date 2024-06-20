import generals.map
import numpy as np

map = np.array([
    [0., 0., 0., 1.,],
    [1., 0., 0., 3.,],
    [2., 0., 0., 1.,],
    [0., 1., 0., 4.,],
])

moves = [
    {'red': np.array([3, 3, 2]), "blue": np.array([1, 3, 2])},
    {'red': np.array([3, 3, 2]), "blue": np.array([1, 3, 2])},
    {'red': np.array([3, 3, 2]), "blue": np.array([1, 3, 2])},
]

generals.map.store_replay(map, moves, "rb")
map, actions = generals.map.load_replay("rb")
print(map)
print(actions)
