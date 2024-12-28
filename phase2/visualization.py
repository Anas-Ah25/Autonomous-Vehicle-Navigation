import pygame

class Visualizer:
    def __init__(self, grid_size, cell_size, screen):
        """
        Initialize the Visualizer with grid and cell dimensions and the Pygame screen.

        Args:
            grid_size (int): The size of the grid (e.g., number of cells in width/height).
            cell_size (int): The size of each cell in pixels.
            screen (pygame.Surface): The Pygame screen where the visualization is drawn.
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.screen = screen
        self.colors = {
            "road": (80, 80, 80),
            "agent": (0, 0, 255),
            "goal": (255, 0, 0),
            "obstacle": (50, 163, 46),
            "hud_text": (255, 255, 255),
        }

    def draw_city_grid(self, maze):
        """
        Draw the city grid based on the maze layout.
        
        Args:
            maze (list of list of int): The maze layout where 1 represents road and 0 represents obstacle.
        """
        for y, row in enumerate(maze):
            for x, cell in enumerate(row):
                # Determine color based on cell value
                color = self.colors["road"] if cell == 1 else self.colors["obstacle"]
                # Draw the cell as a rectangle on the screen
                pygame.draw.rect(
                    self.screen, color,
                    (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                )

    def draw_agent(self, state):
        """
        Draw the agent on the grid.
        
        Args:
            state (tuple of int): The (x, y) position of the agent.
        """
        x, y = state
        # Draw the agent as a smaller rectangle within the cell
        pygame.draw.rect(
            self.screen, self.colors["agent"],
            (x * self.cell_size + 5, y * self.cell_size + 5, self.cell_size - 10, self.cell_size - 10)
        )

    def draw_goal(self, state):
        """
        Draw the goal on the grid.
        
        Args:
            state (tuple of int): The (x, y) position of the goal.
        """
        x, y = state
        # Draw the goal as a rectangle on the screen
        pygame.draw.rect(
            self.screen, self.colors["goal"],
            (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
        )

    def draw_hud(self, steps, reward, epsilon, episode):
        """
        Draw the HUD (Heads-Up Display) with current statistics.
        
        Args:
            steps (int): The number of steps taken.
            reward (float): The current reward.
            epsilon (float): The current epsilon value.
            episode (int): The current episode number.
        """
        font = pygame.font.SysFont(None, 24)
        # Render the HUD text
        hud_text = font.render(
            f"Ep: {episode} | Steps: {steps} | Reward: {reward} | Epsilon: {epsilon:.2f}",
            True, self.colors["hud_text"]
        )
        # Draw the HUD text on the screen
        self.screen.blit(hud_text, (10, 10))

    def display_statistics(self, results):
        """
        Display final results on the game window.
        
        Args:
            results (dict): A dictionary with metrics like total time, avg steps, etc.
        """
        font_title = pygame.font.SysFont(None, 36)
        font_text = pygame.font.SysFont(None, 24)

        # Clear the screen with dark background
        self.screen.fill((30, 30, 30))
        
        # Display Title
        title_text = font_title.render("Learning Statistics", True, (255, 255, 255))
        self.screen.blit(title_text, (self.grid_size * self.cell_size // 2 - 120, 50))
        
        # Display Metrics
        y_offset = 120
        for key, value in results.items():
            text = font_text.render(f"{key}: {value}", True, (255, 255, 255))
            self.screen.blit(text, (50, y_offset))
            y_offset += 40

        # Add Close Button
        close_button = pygame.Rect(self.grid_size * self.cell_size // 2 - 50, y_offset + 50, 100, 40)
        pygame.draw.rect(self.screen, (200, 0, 0), close_button)
        close_text = font_text.render("Close", True, (255, 255, 255))
        self.screen.blit(close_text, (self.grid_size * self.cell_size // 2 - 30, y_offset + 60))

        pygame.display.flip()

        # Wait for user to click the "Close" button
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if close_button.collidepoint(event.pos):
                        waiting = False
