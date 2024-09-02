import generals.utils

map = generals.utils.map_from_generator(
    grid_size=16,
    mountain_density=0.2,
    town_density=0.05,
    n_generals=2,
    general_positions=None,
)
agents = [generals.utils.Player("red"), generals.utils.Player("blue")]

# generals.utils.run_from_replay('test', agents)
generals.utils.run_from_map(map, agents)
