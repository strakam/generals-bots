import pygame

from .properties import Properties


class EventHandler:
    def __init__(self, properties: Properties, from_replay=False):
        """
        Initialize the event handler.

        Args:
            properties: the Properties object
            from_replay: bool, whether the game is from a replay
        """
        self.properties = properties
        self.from_replay = from_replay

    def handle_events(self):
        """
        Handle pygame GUI events
        """
        control_events = {
            "time_change": 0,
        }
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_q
            ):
                pygame.quit()
                quit()

            if event.type == pygame.KEYDOWN and self.from_replay:
                self.__handle_key_controls(event, control_events)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.__handle_mouse_click()

        return control_events


    def __handle_key_controls(self, event, control_events):
        """
        Handle key controls for replay mode.
        Control game speed, pause, and replay frames.
        """
        match event.key:
            # Speed up game right arrow is pressed
            case pygame.K_RIGHT:
                self.properties.game_speed = max(1 / 128, self.properties.game_speed / 2)
            # Slow down game left arrow is pressed
            case pygame.K_LEFT:
                self.properties.game_speed = min(32.0, self.properties.game_speed * 2)
            # Toggle play/pause
            case pygame.K_SPACE:
                self.properties.paused = not self.properties.paused
            case pygame.K_r:
                control_events["restart"] = True
            # Control replay frames
            case pygame.K_h:
                control_events["time_change"] = -1
                self.properties.paused = True
            case pygame.K_l:
                control_events["time_change"] = 1
                self.properties.paused = True


    def __handle_mouse_click(self):
        """
        Handle mouse click event.
        """
        agents = self.properties.game.agents
        agent_fov = self.properties.agent_fov

        x, y = pygame.mouse.get_pos()
        for i, agent in enumerate(agents):
            if self.properties.is_click_on_agents_row(x, y, i):
                agent_fov[agent] = not agent_fov[agent]
                break
