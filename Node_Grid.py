from Libraries import *

'''========================================== Node and Grid Classes ===================================='''

class Node:
    def __init__(self, x, y, goal, g=0):
        self.x = x
        self.y = y
        self.h = self.heuristic(goal)  # heuristic
        self.g = g  # cost
        self.f = self.g + self.h  # total cost (A*)

    def heuristic(self, goal):
        return abs(self.x - goal[0]) + abs(self.y - goal[1])
    def __lt__(self, other):
        return self.f < other.f  # for comparison in priority queue

class Grid:
    def __init__(self, width, height, start, goal):
        self.width = width
        self.height = height
        self.grid = np.zeros((width, height))  # Binary grid (1 = walkable, 0 = blocked)
        self.cost_map = np.ones((width, height))  # Cost map (default = 1)
        self.Moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Possible moves
        self.start = start
        self.goal = goal
        self.pattern()

    def pattern(self):
        # Outer boundaries of the grid
        self.grid[0, :] = 1
        self.grid[:, 0] = 1
        self.grid[-1, :] = 1
        self.grid[:, -1] = 1

        # Horizontal and vertical paths resembling city roads
        for i in range(3, self.width - 3, 3):
            self.grid[i, 1:self.width - 1] = 1
            self.cost_map[i, 1:self.width - 1] = 2  # Example: Higher cost for horizontal paths

        for j in range(3, self.height - 3, 3):
            self.grid[1:self.height - 1, j] = 1
            self.cost_map[1:self.height - 1, j] = 2  # Example: Higher cost for vertical paths

        # Add random obstacles within the grid
        np.random.seed(0)
        obstacles = np.random.randint(1, self.width - 1, size=(int(self.width * self.height * 0.15), 2))
        for (x, y) in obstacles:
            if (x, y) != self.start and (x, y) != self.goal:
                self.grid[x, y] = 0  # Block these cells
                self.cost_map[x, y] = np.inf  # Infinite cost for obstacles


    def get_node(self, x, y):
        return Node(x, y, self.goal,g=self.cost_map[x, y])
    