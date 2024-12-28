import random

class Environment:
    def __init__(self, grid_size, num_obstacles=100):
        self.grid_size = grid_size
        self.goal_state = (grid_size - 1, grid_size - 1)
        self.obstacles = set((random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)) for _ in range(num_obstacles))
        self.obstacles.discard((0, 0))  # Ensure start is not blocked
        self.obstacles.discard(self.goal_state)  # Ensure goal is not blocked

    def get_reward(self, state):
        if state in self.obstacles:
            return -100  # Penalty for hitting an obstacle
        if state == self.goal_state:
            return 100 
        else:
           return -5  # Penalty for taking a step

    def next_state(self, state, action):
        x, y = state
        if action == "UP" and y > 0:
            y -= 1
        elif action == "DOWN" and y < self.grid_size - 1:
            y += 1
        elif action == "LEFT" and x > 0:
            x -= 1
        elif action == "RIGHT" and x < self.grid_size - 1:
            x += 1
        return x, y
