# **Autonomous-Vehicle-Navigation (CSAI 301 Project)**  

This project consists of two phases:  
1. **Path Planning Algorithms Visualization and Comparison**  
2. **Reinforcement Learning for Autonomous Navigation**  

---

## **How It Works**  

The project is divided into two main phases, each focusing on a distinct approach to autonomous navigation.  

### **Phase 1: Path Planning Algorithms**  
1. **Algorithm Selection:** Users can choose from multiple algorithms (*BFS, DFS, UCS, IDS, Greedy, A\*, Hill Climbing, or Simulated Annealing*) via a GUI interface.  
2. **Pathfinding Process:** The selected algorithm runs on a grid layout, with real-time visualization showing the exploration process and node traversal.  
3. **Output:** The final path, along with explored nodes, is displayed as images or videos.  
4. **Analytics:** Performance metrics such as **execution time** and **memory usage** are provided for algorithm comparison.  

### **Phase 2: Reinforcement Learning**  
1. **Training:** The agent learns to navigate an obstacle-filled grid using **Q-Learning**.  
2. **Real-Time Visualization:** The learning process is displayed dynamically using **Pygame**, showing the agent's exploration and decision-making.  
3. **Output:** Metrics such as **success rate**, **average rewards**, and **steps per episode** are logged and exported in a CSV file.  

---

## **Prerequisites**  

- **Python 3.x**  
- **Tkinter** (GUI for Phase 1)  
- **Matplotlib** (Visualizations)  
- **NumPy** (Algorithm Implementation)  
- **Imageio** (Video Rendering for Phase 1)  
- **Pygame** (Visualization for Phase 2)  

---

## **Installation**

1. **Clone the Repository:**  
```bash
git clone https://github.com/Anas-Ah25/Autonomous-Vehicle-Navigation.git
```

2. **Navigate to the Project Directory:**  
```bash
cd Autonomous-Vehicle-Navigation
```

3. **Install Required Dependencies:**  
```bash
pip install -r requirements.txt
```

---

## **Running the Project**

### **Phase 1: Path Planning Algorithms**  

#### **Option 1: Backend (Algorithm Comparison)**  
```bash
python algorithms.py
```
- Generates performance comparison results.  
- Creates `algorithms_results.zip` containing summaries and visualizations.  
- Displays comparison tables in the terminal.  

#### **Option 2: Interactive GUI**  
```bash
python main.py
```
- Launches an interactive GUI.  
- Allows users to select and visualize specific algorithms.  
- Displays execution results and performance metrics.  

### **Phase 2: Reinforcement Learning**  

#### **Running the RL Simulation**  
```bash
python main.py
```
- Starts the Q-Learning agent in the grid environment.  
- Displays real-time training progress and visualization.  

#### **Output Reports:**  
- `learning_report.csv`: Contains detailed metrics per episode, including:  
   - Steps per episode  
   - Total rewards  
   - Success status  

---

## **Output Structure**

```
├── phase1/
│   ├── algorithms.py     # Path planning algorithms
│   ├── gui.py            # GUI implementation
│   ├── node_grid.py      # Grid system setup
│   ├── main.py           # Main GUI application
│   ├── libraries.py      # Required dependencies
│   ├── algorithm_outputs/ 
│       ├── videos/       # Algorithm execution videos
│       ├── images/       # Path visualization images
│       ├── stats/        # Performance statistics

├── phase2/
│   ├── main.py           # Main RL application
│   ├── agent.py          # Q-Learning Agent logic
│   ├── environment.py    # Environment setup
│   ├── visualization.py  # Real-time visualization
│   ├── learning_report.csv  # Training metrics output

├── requirements.txt      # Required dependencies
├── README.md             # Project documentation
```

---

## **Key Features**

- **Phase 1:**  
   - Multiple path planning algorithms implemented: *BFS, DFS, UCS, IDS, Greedy, A\*, Hill Climbing, Simulated Annealing*.  
   - Real-time visualization of algorithm performance.  
   - Performance metrics comparison via analytics reports.  

- **Phase 2:**  
   - Implementation of **Q-Learning** for autonomous navigation.  
   - Dynamic agent visualization using **Pygame**.  
   - Comprehensive training analytics and CSV report generation.  

---

## **Requirements**

- Python 3.x  
- NumPy  
- Matplotlib  
- OpenCV  
- Tkinter  
- Imageio  
- Pygame  

---

## **Contributors**

- **Anas Ahmad**  
- **Ahmed Fouda**  
- **Amin Gamal**  
- **Mohamed Ehab**  
