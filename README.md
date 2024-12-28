# **Autonomous-Vehicle-Navigation (CSAI 301 Project)**  

This project consists of two phases:  
1. **Path Planning Algorithms Visualization and Comparison**  
2. **Reinforcement Learning for Autonomous Navigation**  

---

## **📋 Phase 1: Path Planning Algorithms**

### **Overview**  
An interactive system for **visualizing and comparing different pathfinding algorithms** through a GUI interface. The system allows users to select specific algorithms and observe their behavior in real-time on a grid-based environment.

---

### **🛠 Setup & Installation**

1. **Install Required Dependencies:**  
```bash
pip install -r requirements.txt
```

2. **Run Files in the Following Order (if issues arise):**  
```bash
python libraries.py
python node_grid.py
python algorithms.py
python gui.py
python main.py
```

---

### **🔄 Running Options**

#### **Option 1: Algorithm Comparison (Backend)**  
Run `algorithms.py` to:  
- Generate performance comparison of all algorithms.  
- Create `algorithms_results.zip` containing summaries and visualizations.  
- Display a comparison table in the terminal.

#### **Option 2: Interactive GUI (Frontend)**  
Run `main.py` to:  
- Launch the **interactive GUI**.  
- Select specific algorithms to visualize.  
- View real-time execution results.  
- Compare selected algorithms' performance.

---

### **📂 Output Structure**
- `algorithm_outputs/`: Main output directory containing:  
   - **Algorithm execution videos**  
   - **Path visualization images**  
   - **Exploration process visualizations**  
   - **Performance statistics**  

---

### **🎯 Features**
- Multiple algorithm implementations (*BFS, DFS, UCS, IDS, Greedy, A\*, Hill Climbing, Simulated Annealing*).  
- Real-time visualization of algorithm execution.  
- Comparison of performance metrics (e.g., execution time, explored nodes).  
- User-friendly **interactive GUI interface**.

---
---
## **📋 Phase 2: Reinforcement Learning Navigation**

### **Overview**  
An implementation of **Q-Learning-based Reinforcement Learning (RL)** for autonomous navigation in a simulated grid environment. The agent learns to navigate efficiently while avoiding obstacles and reaching a goal state.

---

### **🛠 Setup & Installation**

1. **Install Required Dependencies:**  
```bash
pip install -r requirements.txt
```

2. **Run the Simulation:**  
```bash
python main.py
```

3. **Observe Real-Time Visualization:**  
- The agent's learning behavior will be displayed on the screen.  
- Metrics will update live during each episode.

---

### **📊 Output**
- **`learning_report.csv`:** A detailed report containing:  
   - **Episode Number**  
   - **Steps Taken per Episode**  
   - **Total Rewards Earned**  
   - **Success/Failure Status**  

- **Statistics Screen:** After training, a final screen displays:  
   - Total Episodes  
   - Average Steps per Episode  
   - Average Reward per Episode  
   - Success Rate  
   - Total Training Time  

---

### **🎯 Features**
- Real-time **visualization of the learning process** using Pygame.  
- Adaptive learning parameters (`alpha`, `gamma`, `epsilon`).  
- Post-training analytics through **`learning_report.csv`**.  
- Visualization of agent behavior and policy convergence.

---

## **🏗 Project Structure**

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

## **🔍 Requirements**

- **Python 3.x**  
- **NumPy**  
- **Matplotlib**  
- **OpenCV**  
- **Tkinter**  
- **Imageio**  
- **Pygame**  

---

## **👥 Contributors**

- **Anas Ahmad**  
- **Ahmed Fouda**  
- **Amin Gamal**  
- **Mohamed Ehab**  

---

## **📝 Notes**

- **Phase 1:** Focused on comparing and visualizing **pathfinding algorithms** with static search strategies.  
- **Phase 2:** Explored **dynamic learning behavior** through **Q-Learning** in a grid environment.  
- Performance analytics and visualizations are provided for both phases.  
