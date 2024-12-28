import pygame
import csv
import time
from agent import Agent
from enviroment import Environment
from visualization import Visualizer

pygame.init()
''' ========================================================================================= '''
# --------------------------------  Configuration parameters ---------------------------------
''' ========================================================================================= '''

GRID_SIZE = 10      # Grid dimensions (10x10)
CELL_SIZE = 40      # cells size
EPISODES = 65      #  training episodes
STEPS_PER_EPISODE = 70  # Maximum steps per episode
ALPHA = 0.1         # learning rate
GAMMA = 0.9         # penalty factor
EPSILON = 0.3       # exploration rate
EPSILON_DECAY = 0.99  # Decay rate per episode
MIN_EPSILON = 0.1   # Minimum exploration rate
FPS = 60            # visualization speed

# ------------ Initialize --------------- 
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Car Path Learning Visualization")

env = Environment(GRID_SIZE)
agent = Agent(GRID_SIZE, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON)
visualizer = Visualizer(GRID_SIZE, CELL_SIZE, screen)

# ---------- Part 1 ( Training ) -----------------------
start_time = time.time()
report_data = []

for episode in range(EPISODES):
    state = (0, 0)
    total_reward = 0
    steps = 0
    goal_reached = False

    for step in range(STEPS_PER_EPISODE):
        steps += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # ------ actions taking ------ 
        action = agent.choose_action(state)
        new_state = env.next_state(state, action)
        reward = env.get_reward(new_state)
        total_reward += reward
        agent.update_q_value(state, action, reward, new_state)

        #  --------------- visualiztion ------------------
        screen.fill((50, 50, 50))
        visualizer.draw_city_grid(env.maze)
        visualizer.draw_goal(env.goal_state)
        visualizer.draw_agent(new_state)
        visualizer.draw_hud(steps, total_reward, agent.epsilon, episode + 1)
        pygame.display.flip()
        pygame.time.delay(int(1000 / FPS))

        # --------------- print on reaching goal ------------ 
        if new_state == env.goal_state:
            print(f"Goal reached in Episode {episode + 1} after {steps} steps with Total Reward: {total_reward}")
            goal_reached = True
            break

        state = new_state

    agent.decay_epsilon(EPSILON_DECAY, MIN_EPSILON)
    report_data.append([episode + 1, steps, total_reward, goal_reached])

# ---------------------- Part 2 (statisitics window) ----------------------------
total_time = round(time.time() - start_time, 2)
results = {
    "Total Episodes": EPISODES,
    "Average Steps": round(sum(row[1] for row in report_data) / len(report_data), 2),
    "Average Reward": round(sum(row[2] for row in report_data) / len(report_data), 2),
    "Success Rate (%)": round(sum(1 for row in report_data if row[3]) / len(report_data) * 100, 2),
    "Total Training Time (s)": total_time,
}

visualizer.display_statistics(results)

# --------------- Report as CSV ------------------------
with open('learning_report.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['Episode', 'Steps', 'Total Reward', 'Goal Reached'])
    writer.writerows(report_data)
    writer.writerow([])
    writer.writerow(['Total Time (s)', total_time])
    writer.writerow(['Average Steps', results["Average Steps"]])
    writer.writerow(['Average Reward', results["Average Reward"]])
    writer.writerow(['Success Rate (%)', results["Success Rate (%)"]])

print(f"Learning report saved as 'learning_report.csv'. Total Time: {total_time} seconds.")

pygame.quit()
