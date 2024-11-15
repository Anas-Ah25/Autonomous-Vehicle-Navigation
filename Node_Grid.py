from Libraries import *
# ---------------- Node and Grid Classes ----------------
class Node:
    def __init__(self, x, y, goal, g=0):
        self.x = x
        self.y = y
        self.h = self.heuristic(goal)  # heuristic
        self.g = g  # cost
        self.f = self.g + self.h  # total cost (A*)

    def heuristic(self, goal):
        return abs(self.x - goal[0]) + abs(self.y - goal[1])


class Grid:
    def __init__(self, width, height, start, goal):
        self.width = width
        self.height = height
        self.grid = np.zeros((width, height))
        self.Moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.start = start
        self.goal = goal
        self.pattern()

    def pattern(self):  # This is grid pattern

        # Outer boundaries of the grid
        self.grid[0, :] = 1
        self.grid[:, 0] = 1
        self.grid[-1, :] = 1
        self.grid[:, -1] = 1

        # Horizontal and vertical paths resembling city roads
        for i in range(3, self.width - 3, 3):
            self.grid[i, 1:self.width - 1] = 1

        for j in range(3, self.height - 3, 3):
            self.grid[1:self.height - 1, j] = 1

        # Add random obstacles within the grid
        np.random.seed(0)
        obstacles = np.random.randint(1, self.width - 1, size=(int(self.width * self.height * 0.15), 2))
        for (x, y) in obstacles:
            if (x, y) != self.start and (x, y) != self.goal:
                self.grid[x, y] = 0  # Block these cells

        # Central area with complex paths
        self.grid[10:15, 5:20] = 1
        self.grid[5:10, 10:15] = 1
        self.grid[15:20, 10:15] = 1
        self.grid[10:15, 10:15] = 0

        # Additional paths to increase the grid complexity
        self.grid[5, 5:10] = 1
        self.grid[10:15, 3] = 1
        self.grid[15, 15:20] = 1
        self.grid[20, 5:10] = 1
        self.grid[20, 15:25] = 1
        self.grid[5:7, 24] = 0

    def get_node(self, x, y):
        return Node(x, y, self.goal)
    