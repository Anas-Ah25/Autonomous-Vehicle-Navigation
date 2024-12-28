import pygame
from agent import Agent
from environment import Environment
from visualization import Visualizer

# Pygame initialization
pygame.init()

# Screen settings
GRID_SIZE = 25  # 25x25 grid
CELL_SIZE = 20  # Each cell is 20x20 pixels
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Q-Learning Visualization")

# Initialize Environment, Agent, and Visualizer
env = Environment(GRID_SIZE)
agent = Agent(GRID_SIZE)
visualizer = Visualizer(GRID_SIZE, CELL_SIZE, screen)

# Metrics for performance tracking
steps_per_episode = []
cumulative_rewards = []

# Main Q-learning loop
for episode in range(300):
    state = (0, 0)
    total_reward = 0
    steps = 0

    for step in range(100):
        steps += 1

        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Agent takes an action
        action = agent.choose_action(state)
        new_state = env.next_state(state, action)
        reward = env.get_reward(new_state)
        total_reward += reward

        # Update Q-value
        agent.update_q_value(state, action, reward, new_state)

        # Visualization
        screen.fill((255, 255, 255))
        visualizer.draw_grid()
        visualizer.draw_agent(state, (0, 0, 255))  # Blue agent
        visualizer.draw_agent(env.goal_state, (0, 255, 0))  # Green goal
        visualizer.draw_obstacles(env.obstacles)
        visualizer.draw_agent(new_state, (255, 0, 0))  # Red new state
        visualizer.draw_q_values(agent.q_table)
        pygame.display.flip()
        pygame.time.delay(10)

        if new_state == env.goal_state:
            visualizer.display_goal_reached()
            break

        state = new_state

    steps_per_episode.append(steps)
    cumulative_rewards.append(total_reward)
    agent.decay_epsilon()

# Plot performance metrics
visualizer.plot_performance(steps_per_episode, cumulative_rewards)
visualizer.plot_heatmap(agent.q_table)

pygame.quit()
