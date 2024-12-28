class Environment:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.goal_state = (grid_size-1, grid_size-1)

        # ------------ manually define the maze, to simulate we are in a city and we have streets ---- 
        # random obnstacles was too noisy and not meaningful 
        self.maze = [
            [1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1, 0, 0, 1],
            [0, 1, 0, 0, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        ]
        
        self.obstacles = self.get_obstacles()
    
    # ------------- obstacles -----------------
    def get_obstacles(self):
        obstacles = set()
        for y, row in enumerate(self.maze):
            for x, cell in enumerate(row):
                if cell == 0:
                    obstacles.add((x, y))
        return obstacles

    def get_reward(self, state):
        if state in self.obstacles:
            return -100  # penalty for hitting an obstacle
        return 100 if state == self.goal_state else -1

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
        
        if (x, y) in self.obstacles:
            return state  # stay in place if the move hits an obstacle
        
        return x, y
