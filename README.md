# 🚗 Autonomous-Vehicle-Navigation

This project provides a **graphical user interface (GUI)** to visualize and compare different pathfinding algorithms such as **BFS**, **DFS**, **IDS**, and **Simulated Annealing**. 🎯 The GUI enables users to select an algorithm, observe its step-by-step execution, and analyze its performance.

---

## 📋 Table of Contents
- [🛠 Features](#-features)
- [📂 Project Structure](#-project-structure)
- [🚀 How It Works](#-how-it-works)
- [⚙ Prerequisites](#-prerequisites)
- [📦 Installation](#-installation)
- [🏃 Usage](#-usage)
- [📜 Code Explanation](#-code-explanation)
- [🖥 GUI Components](#-gui-components)
- [📈 Analytics](#-analytics)
- [💡 Future Enhancements](#-future-enhancements)

---

## 🛠 Features
- **Interactive GUI**:
  - Select algorithms via a user-friendly interface. 🖱️
  - Visualize explored nodes and final paths. 🧩
  - Switch between algorithms without restarting the application. 🔁

- **Real-time Feedback**:
  - Observe exploration and pathfinding videos directly in the GUI. 🎥
  - View detailed analytics for performance comparison. 📊

- **Modular Design**:
  - Each algorithm is implemented independently for extensibility. 🛠️

---

## 📂 Project Structure
├── phase1/
│   ├── algorithms.py   # Contains functions like bfs, dfs, ucs, etc.
│   ├── gui.py          # Handles GUI-related functionality
│   ├── node_grid.py    # Contains node and grid-related functionality
│   ├── main.py         # Main script for running phase 1
|   ├── libraries.py    # Contains all the needed libraries
|   ├── astar_vs_greedy.py   # Compare A* and Greedy performance
├── phase2/
│   ├── agent.py        # Contains functions like choose_action, update_q_value, etc.
│   ├── environment.py  # Contains functions like get_obstacles, get_reward, etc.
│   ├── visualization.py # Contains functions like draw_city_grid, draw_agent, etc.
│   ├── main.py         # Main script for running phase 2
├── README.md           # Project overview and documentation
├── requirements.txt    # Dependencies for the project

---

## 🚀 How It Works
The project consists of the following key steps:
1. **Algorithm Selection**: Users can choose from different algorithms (BFS, DFS, UCS, IDS, Gready, A*, Hill Climbing, or Simulated Annealing) via the GUI.
2. **Pathfinding Process**: Once an algorithm is selected, the program runs the chosen pathfinding algorithm on a grid. The grid is displayed in real-time, and the algorithm's progress is visualized as it explores nodes.
3. **Output**: Once the algorithm completes, the final path and explored nodes are displayed as images or videos.
4. **Analytics**: The performance metrics like execution time and memory usage are displayed to compare the efficiency of different algorithms.

---

## ⚙ Prerequisites
- Python 3.x
- Tkinter (for GUI)
- Matplotlib (for visualizations)
- NumPy (for algorithm implementation)
- Imageio
- pygame

---

## 📦 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Anas-Ah25/Autonomous-Vehicle-Navigation.git
   ```
2. Navigate to the project directory:
```bash
cd Autonomous-Vehicle-Navigation
```
3.Install required dependencies:
```bash
pip install -r requirements.txt
```
## 🏃 Usage
1.Launch the application by running:
```bash
python main.py
```
2.Select an algorithm from the list.
3.Watch the algorithm explore and find the path videos and the final path, exploration path images.
4.View the results and compare performance metrics.


## 📜 Code Explanation
The program consists of various components that handle the user interface and the algorithm execution:

- **Algorithm Selection GUI**: Allows users to select the algorithm they wish to run.
- **Loading Screen GUI**: Displays a loading screen while the algorithm is running.
- **Result GUI**: Displays the final output, including videos and images of the pathfinding process.
- **Analytics**: Displays the performance metrics like execution time and memory usage.

---

## 🖥 GUI Components
The project includes several GUI components:

- **Algorithm Selection Window**: Allows users to select the algorithm to use.
- **Loading Screen**: Displays a message while the algorithm is running.
- **Result Screen**: Displays the results of the algorithm, including exploration videos, path images, and final videos.

---

## 📈 Analytics
The analytics page displays performance metrics such as execution time, memory usage, and comparisons between different algorithms.

- **Execution Time**: The time taken by each algorithm to complete the pathfinding process.
- **Memory Usage**: The maximum memory usage during the execution of the algorithm.

---

## 💡 Future Enhancements
- **Support for More Algorithms**: Adding additional pathfinding algorithms like A* and Dijkstra.
- **Advanced Analytics**: Provide deeper insights into the algorithm’s performance, such as step-by-step memory profiling.
- **User Customization**: Allow users to modify the grid size, obstacle placement, and starting/ending points.

