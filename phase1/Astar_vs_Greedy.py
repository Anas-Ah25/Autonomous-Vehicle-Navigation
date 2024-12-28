from Libraries import *

'''================================================================================================'''
'''========================== compare A* and Greedy Best-First Search ============================='''
'''================================================================================================'''
# Testing version with different grid initialization and structure
import math
import numpy as np
import time
import pandas as pd
import heapq
from tabulate import tabulate


class Node:
    def __init__(self, x, y, goal, heuristic_type="manhattan", g=0, parent=None):
        self.x = x
        self.y = y
        self.heuristic_type = heuristic_type
        self.h = self.heuristic(goal)
        self.g = g  # Cumulative cost from the start node
        self.f = self.g + self.h
        self.parent = parent  # To reconstruct the path

    def heuristic(self, goal):
        if self.heuristic_type == "manhattan":
            return abs(self.x - goal[0]) + abs(self.y - goal[1])
        elif self.heuristic_type == "euclidean":
            return math.hypot(self.x - goal[0], self.y - goal[1])
        return 0

    def __lt__(self, other):
        return self.f < other.f  # For priority queue comparisons

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


class Grid:
    def __init__(self, width, height, start, goal):
        self.width = width
        self.height = height
        self.grid = np.zeros((width, height))  # 0, walkable cell and 1 is obstacle
        self.cost_map = np.ones((width, height))  # Initialize all costs to 1
        self.Moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Possible moves
        self.start = start
        self.goal = goal
        self.pattern()

    def pattern(self):
        # Create a simple grid with some obstacles and random costs
        np.random.seed(0)
        num_obstacles = int(self.width * self.height * 0.15)
        obstacles = np.random.randint(1, self.width - 1, size=(num_obstacles, 2))
        for (x, y) in obstacles:
            if (x, y) != self.start and (x, y) != self.goal:
                self.grid[x, y] = 1  # Mark as obstacle
                self.cost_map[x, y] = np.inf  # Infinite cost for obstacles

        # Randomly assign higher costs (2 and 3) to some walkable cells
        num_high_cost_cells = int(self.width * self.height * 0.1)
        high_cost_cells = np.random.randint(1, self.width - 1, size=(num_high_cost_cells, 2))
        for (x, y) in high_cost_cells:
            if (x, y) != self.start and (x, y) != self.goal and self.grid[x, y] == 0:
                self.cost_map[x, y] = np.random.choice([2, 3])  # Assign cost 2 or 3

    def is_valid(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height and self.grid[x, y] == 0

    def get_neighbors(self, node):
        neighbors = []
        for move in self.Moves:
            nx, ny = node.x + move[0], node.y + move[1]
            if self.is_valid(nx, ny):
                neighbors.append((nx, ny))
        return neighbors

    def get_node(self, x, y, heuristic_type="manhattan"):
        return Node(x, y, self.goal, heuristic_type)


class Algorithm(Grid):
    def __init__(self, width, height, start, goal, heuristic_type="manhattan"):
        super().__init__(width, height, start, goal)
        self.heuristic_type = heuristic_type

    def a_star(self):
        open_list = []
        heapq.heapify(open_list)
        start_node = self.get_node(*self.start, self.heuristic_type)
        start_node.g = 0
        start_node.f = start_node.h
        heapq.heappush(open_list, start_node)
        closed_set = set()
        total_cost = 0

        while open_list:
            current = heapq.heappop(open_list)
            if (current.x, current.y) in closed_set:
                continue
            closed_set.add((current.x, current.y))

            if (current.x, current.y) == self.goal:
                # Reconstruct the path
                path = []
                while current:
                    path.append((current.x, current.y))
                    current = current.parent
                path.reverse()
                final_path_cost = sum(self.cost_map[x, y] for x, y in path)
                return path, "A*", len(closed_set), len(open_list), total_cost, final_path_cost, time.time()

            for nx, ny in self.get_neighbors(current):
                if (nx, ny) in closed_set:
                    continue
                move_cost = self.cost_map[nx, ny]
                tentative_g = current.g + move_cost
                neighbor = Node(nx, ny, self.goal, self.heuristic_type)
                neighbor.g = tentative_g
                neighbor.f = neighbor.g + neighbor.h
                neighbor.parent = current

                # Check if neighbor is in open_list with a higher g value
                in_open = False
                for node in open_list:
                    if neighbor == node and neighbor.g >= node.g:
                        in_open = True
                        break
                if not in_open:
                    heapq.heappush(open_list, neighbor)

        return [], "A*", len(closed_set), len(open_list), total_cost, 0, time.time()

    def greedy_algorithm(self):
        open_list = []
        heapq.heapify(open_list)
        start_node = self.get_node(*self.start, self.heuristic_type)
        start_node.g = 0
        start_node.f = start_node.h  # For greedy, f = h
        heapq.heappush(open_list, start_node)
        closed_set = set()
        total_cost = 0

        while open_list:
            current = heapq.heappop(open_list)
            if (current.x, current.y) in closed_set:
                continue
            closed_set.add((current.x, current.y))

            if (current.x, current.y) == self.goal:
                # Reconstruct the path
                path = []
                while current:
                    path.append((current.x, current.y))
                    current = current.parent
                path.reverse()
                final_path_cost = sum(self.cost_map[x, y] for x, y in path)
                return path, "Greedy", len(closed_set), len(open_list), total_cost, final_path_cost, time.time()

            for nx, ny in self.get_neighbors(current):
                if (nx, ny) in closed_set:
                    continue
                neighbor = Node(nx, ny, self.goal, self.heuristic_type)
                neighbor.g = current.g + self.cost_map[nx, ny]  # Accumulate cost
                neighbor.f = neighbor.h  # For greedy, f = h
                neighbor.parent = current

                # Check if neighbor is in open_list
                in_open = False
                for node in open_list:
                    if neighbor == node:
                        in_open = True
                        break
                if not in_open:
                    heapq.heappush(open_list, neighbor)

        return [], "Greedy", len(closed_set), len(open_list), total_cost, 0, time.time()


# Comparison function
def compare_a_star_and_greedy(width, height, start, goal):
    results = []
    for heuristic in ["manhattan", "euclidean"]:
        algo_instance = Algorithm(width, height, start, goal, heuristic)

        # A* Search
        a_star_path, a_star_name, a_star_closed, a_star_open, a_star_cost, a_star_final, a_star_time = algo_instance.a_star()
        results.append({
            "Algorithm": "A*",
            "Heuristic": heuristic,
            "Path Length": len(a_star_path),
            "Explored Nodes": a_star_closed,
            "Open List Size": a_star_open,
            "Final Path Cost": a_star_final,
        })

        # Greedy Best-First Search
        greedy_path, greedy_name, greedy_closed, greedy_open, greedy_cost, greedy_final, greedy_time = algo_instance.greedy_algorithm()
        results.append({
            "Algorithm": "Greedy",
            "Heuristic": heuristic,
            "Path Length": len(greedy_path),
            "Explored Nodes": greedy_closed,
            "Open List Size": greedy_open,
            "Final Path Cost": greedy_final,
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    width, height = 25, 25
    start = (0, 0)
    goal = (24, 24)
    results_df = compare_a_star_and_greedy(width, height, start, goal)
    print(tabulate(results_df.values, headers=results_df.columns, tablefmt='pretty'))
