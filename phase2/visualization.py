import pygame
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Visualizer:
    def __init__(self, grid_size, cell_size, screen):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.screen = screen
        self.white = (255, 255, 255)
        self.gray = (200, 200, 200)
        self.green = (0, 255, 0)
        self.blue = (0, 0, 255)
        self.orange = (255, 165, 0)
        self.grey = (128, 128, 128)
        
        self.episode_font = pygame.font.SysFont(None, 24)

    def draw_grid(self):
        for x in range(0, self.grid_size * self.cell_size, self.cell_size):
            pygame.draw.line(self.screen, self.gray, (x, 0), (x, self.grid_size * self.cell_size))
        for y in range(0, self.grid_size * self.cell_size, self.cell_size):
            pygame.draw.line(self.screen, self.gray, (0, y), (self.grid_size * self.cell_size, y))

    def draw_agent(self, state, color):
        x, y = state
        pygame.draw.rect(self.screen, color, (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))

    def draw_q_values(self, q_table):
        font = pygame.font.SysFont(None, 14)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                state = (x, y)
                for i, action in enumerate(["UP", "DOWN", "LEFT", "RIGHT"]):
                    q_value = q_table[state][action]
                    text = font.render(f"{q_value:.1f}", True, self.grey)
                    offset = [(self.cell_size // 2, 5), (self.cell_size // 2, self.cell_size - 15), (5, self.cell_size // 2), (self.cell_size - 20, self.cell_size // 2)]
                    pos = (x * self.cell_size + offset[i][0], y * self.cell_size + offset[i][1])
                    self.screen.blit(text, pos)

    def draw_obstacles(self, obstacles):
        for obs in obstacles:
            self.draw_agent(obs, self.orange)

    def display_goal_reached(self):
        font = pygame.font.SysFont(None, 48)
        text = font.render("Goal Reached!", True, self.green)
        self.screen.blit(text, (self.grid_size * self.cell_size // 2 - 100, self.grid_size * self.cell_size // 2 - 20))
        pygame.display.flip()
        pygame.time.delay(1000)

    def plot_heatmap(self, q_table):
        state_values = np.zeros((self.grid_size, self.grid_size))
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                state = (x, y)
                state_values[x, y] = max(q_table[state].values())
        plt.figure(figsize=(10, 8))
        sns.heatmap(state_values, cmap="coolwarm", cbar=True)
        plt.title("State Value Heatmap")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.show()

    def plot_performance(self, steps_per_episode, cumulative_rewards):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(cumulative_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Cumulative Reward")
        plt.title("Cumulative Rewards Over Episodes")
        plt.tight_layout()
        plt.show()

    def draw_episode_number(self, episode):
        text = self.episode_font.render(f"Episode: {episode}", True, self.blue)
        text_rect = text.get_rect()
        padding = 10
        text_rect.topright = (self.grid_size * self.cell_size - padding, padding)
        self.screen.blit(text, text_rect)