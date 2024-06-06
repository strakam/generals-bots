import pygame


display_width = 800
display_height = 800

class Visualizer():
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((display_width, display_height))
        self.clock = pygame.time.Clock()

    def draw_grid(self, grid):
        self.screen.fill("purple")
        pygame.draw.line(self.screen, "red", (20, 20), (100, 100))
        print('no shit')
        pygame.display.flip()
