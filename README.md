# 🚗 Autonomous-Vehicle-Navigation (CSAI 301 Project)

This project consists of two phases:
1. Path Planning Algorithms Visualization and Comparison
2. Reinforcement Learning for Autonomous Navigation


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

2. Run files in the following order:
```bash
python libraries.py
python node_grid.py
python algorithms.py
python Gui.py
python main.py
```

### 🔄 Running Options

#### Option 1: Algorithm Comparison (Backend)
Run `algorithms.py` to:
- Generate performance comparison of all algorithms
- Create `algorithms_results.zip` containing summary and visualizations
- Display comparison table in terminal

#### Option 2: Interactive GUI (Frontend)
Run `main.py` to:
- Launch interactive GUI
- Select specific algorithms to visualize
- View real-time execution results
- Compare selected algorithms' performance

### 📂 Output Structure
- `algorithm_outputs/`: Main output directory containing:
  - Algorithm execution videos
  - Path visualization images
  - Exploration process visualizations
  - Performance statistics

### 🎯 Features
- Multiple algorithm implementations (BFS, DFS, UCS, IDS, Greedy, A*, Hill Climbing, Simulated Annealing)
- Real-time visualization
- Performance metrics comparison
- Interactive GUI interface

---

## 📋 Phase 2: Reinforcement Learning Navigation

### Overview
An implementation of reinforcement learning for autonomous navigation in a simulated environment.

### 🎮 Running the Simulation
1. Execute the main script:
```bash
python main.py
```

2. Observe the agent learning process in real-time

### 📊 Output
- `learning_report.csv`: Contains detailed statistics about the agent's learning process including:
  - Episode information
  - Rewards earned
  - Learning progress
  - Performance metrics

### 🎯 Features
- Real-time agent visualization
- Learning progress tracking
- Performance analytics
- Environment interaction simulation

---

## 🏗 Project Structure
```
├── phase1/
│   ├── algorithms.py   # Path planning algorithms
│   ├── gui.py         # GUI implementation
│   ├── node_grid.py   # Grid system
│   ├── main.py        # Main GUI application
│   ├── libraries.py   # Required dependencies
├── phase2/
│   ├── main.py        # main running app
│   ├── agent.py       # Agent implementation
│   ├── environment.py # Environment setup
│   ├── visualization.py # gui
```

## 🔍 Requirements
- Python 3.x
- NumPy
- Matplotlib
- OpenCV
- Tkinter
- Imageio
- Pygame (for Phase 2)

## 👥 Contributors
- Anas Ahmad 
- Ahmed Fouda 
- Amin Gamal
- Mohamed ehab

## 📝 Note
This project is part of the CSAI 301 course curriculum, focusing on implementing and comparing different approaches to autonomous navigation.

