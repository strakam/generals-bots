import pygame



class Visualizer():
    square_size = 40
    def __init__(self, grid_size=20):
        pygame.init()
        self.grid_size = grid_size
        self.window_size = self.square_size * grid_size
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        self.clock = pygame.time.Clock()

        self.initial_draw()

    def draw_grid(self, grid):
        pygame.draw.line(self.screen, "red", (20, 20), (100, 100))
        print('no shit')
        pygame.display.flip()

    def initial_draw(self):
        self.screen.fill("gray")
        for i in range(1, self.grid_size):
            pygame.draw.line(self.screen, "black", (self.square_size*i, 0), (self.square_size*i, self.window_size), 2)
            pygame.draw.line(self.screen, "black", (0, self.square_size*i), (self.window_size, self.square_size*i), 2)

