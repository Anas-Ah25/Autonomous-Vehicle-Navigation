import random
from collections import defaultdict

class Agent:
    def __init__(self, grid_size, alpha=0.1, gamma=0.9, epsilon=0.3):
        self.grid_size = grid_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: {a: 0.0 for a in ["UP", "DOWN", "LEFT", "RIGHT"]})

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(["UP", "DOWN", "LEFT", "RIGHT"])
        else:
            return max(self.q_table[state], key=self.q_table[state].get)

    def update_q_value(self, state, action, reward, next_state):
        best_future_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        self.q_table[state][action] += self.alpha * (reward + self.gamma * self.q_table[next_state][best_future_action] - self.q_table[state][action])

    def decay_epsilon(self, decay_rate=0.99, min_epsilon=0.1):
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
