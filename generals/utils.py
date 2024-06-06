import pygame


class Visualizer():
    square_size = 40
    def __init__(self, grid_size):
        pygame.init()
        self.grid_size = grid_size
        self.window_size = self.square_size * grid_size
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        self.clock = pygame.time.Clock()

    def draw_grid(self, grid):
        self.screen.fill("gray")

        # draw terrain
        terrain_indices = grid.get_channel("terrain")
        for i, j in terrain_indices:
            pygame.draw.rect(
                self.screen,
                "green",
                (i*self.square_size, j*self.square_size, self.square_size, self.square_size)
            )

        for i in range(1, self.grid_size):
            pygame.draw.line(self.screen, "black", (self.square_size*i, 0), (self.square_size*i, self.window_size), 2)
            pygame.draw.line(self.screen, "black", (0, self.square_size*i), (self.window_size, self.square_size*i), 2)

        pygame.display.flip()

