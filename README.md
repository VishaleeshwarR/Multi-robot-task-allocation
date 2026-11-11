# Multi-Robot Task Allocation in Dynamic Environments

Implementation and Webots simulation of the game-theoretic multi-robot task allocation framework from:

> **"Multi-Robot Task Allocation Games in Dynamically Changing Environments"**  
> *Shinkyu Park, Yaofeng Desmond Zhong, and Naomi Ehrich Leonard*  
> IEEE International Conference on Robotics and Automation (ICRA 2021)  
> [IEEE Xplore](https://ieeexplore.ieee.org/document/9561849) | [arXiv:2009.05193](https://arxiv.org/abs/2009.05193)

Repository: [https://github.com/VishaleeshwarR/Multi-robot-task-allocation.git](https://github.com/VishaleeshwarR/Multi-robot-task-allocation.git)

---

## Workspace Modifications & Key Enhancements

This repository contains significant enhancements to transform the conceptual framework into a fully functioning, high-performance robotic simulation in Webots alongside analytical benchmark validation.

### 1. Dynamic Experiment Engine (`controllers/supervisor_controller/supervisor_controller.py`)
- **Automated Experiment Execution**: Implemented dynamic scenarios for all 4 experiments from the ICRA 2021 paper:
  - **Experiment 1 (Temporary Surge)**: Workload rate $w_1$ surges from $0.5 \to 5.0$ objects/s during $[20\text{s}, 60\text{s}]$ (demo) or $[500\text{s}, 600\text{s}]$ (paper), testing agile reallocation and return to baseline.
  - **Experiment 2 (Permanent Growth Increase)**: $w_1$ shifts permanently from $0.5 \to 5.0$ objects/s starting at $t=20\text{s}$ (demo) or $t=500\text{s}$ (paper), testing asymptotic convergence to a new equilibrium.
  - **Experiment 3 (Robot Failures & Attrition)**: 15 out of 40 robots abruptly shut down during $[20\text{s}, 60\text{s}]$ (demo) or $[500\text{s}, 1500\text{s}]$ (paper) and subsequently recover, verifying fleet resilience.
  - **Experiment 4 (Payoff Weight $\nu$)**: Evaluates predictive model-based allocation ($\nu > 0$) vs. reactive model-free allocation ($\nu = 0$) under asymmetric growth $W = [1.5, 0.5, 0.5, 0.5]$.
- **Physical Proximity Trash Collection**: Replaced synthetic depletion with true 3D spatial collection ($r \le 0.35\,\text{m}$), removing trash solids from the Webots tree and updating resource levels $q$ in real-time.
- **Physics Engine Optimization**: Streamlined visual trash Solid representations (removing burdensome collision meshes) and capped active markers per patch to 25, eliminating ODE physics slowdowns with 40 robots.
- **Automated Telemetry & Plotting**: Automatically captures state trajectories ($q_i, x_i, w_i$, active robots) every second, exports data to `results/webots_exp{N}_data.csv`, and renders publication-quality matplotlib plots to `results/webots_exp{N}_measured.png`.
- **Headless Clean Exit**: Supports automated headless batch runs with `simulationQuit(0)` upon reaching target simulation duration.

### 2. Autonomous Robot Controller (`controllers/robot_controller/robot_controller.py`)
- **RF Communication Architecture**: Set up dedicated Webots wireless channels (Channel 1: supervisor broadcast receiver, Channel 2: robot emitter), packing and unpacking telemetry without cross-talk.
- **Closed-Loop Target Seeking & Patrolling**: E-puck robots continuously track nearest trash items in their assigned patch; when cleared, they perform bounded exploratory patrolling.
- **Decisive Obstacle & Inter-Robot Collision Avoidance**: Multi-sensor infrared obstacle detection with decisive turn behaviors and motor velocity clamping $[-6.28, 6.28]\,\text{rad/s}$ to prevent wedging or stalls.
- **Protected Poisson Task Revision**: Poisson clock revises task assignment only during `STATE_FORAGING`, preventing disruptions while robots are transitioning across the arena or maneuvering around obstacles.
- **Adaptive Revision Scaling**: Dynamically scales $\rho$ when resource counts are low to preserve responsive task switching without stagnation.
- **Attrition Handling**: Responds to supervisor fleet count broadcasts by idling inactive robot IDs during Experiment 3.

### 3. Cross-Platform Simulation Launcher (`run.py`)
- Replaced a macOS-only AppleScript launcher with a robust, cross-platform CLI tool supporting Linux, macOS, and Windows.
- Automatic binary discovery (`which webots`, `/snap/bin/webots`, `/usr/bin/webots`, standard paths).
- Flexible CLI flags for experiment selection (`-e`), accelerated demo timelines (`-q`), headless fast-mode batch execution (`-b`), and custom payoff weights (`--nu`).

### 4. Standalone Analytical ODE Solver (`experiments/run_experiments.py`)
- Provides an independent numerical verification suite utilizing `scipy.integrate.solve_ivp` (RK45).
- Computes theoretical trajectories for Equations (2), (3), (6), and (7), allowing direct 1:1 validation between analytical predictions and Webots 3D physics execution.
- Generates comparative plots saved to `results/`.

### 5. World Performance Optimization (`worlds/trash_collection.wbt`)
- Increased simulation basic timestep to 32 ms and allocated 8 threads (`optimalThreadCount 8`) for high multi-core throughput with 40 e-puck agents.

---

## Core Algorithms & Mathematical Framework

### 1. Resource Dynamics (Equation 2)
$$\dot{q}_i(t) = -F_i(q_i(t), x_i(t)) + w_i(t)$$

Where:
- $q_i(t) \ge 0$: Resource level (trash backlog) in area $i$.
- $w_i(t) > 0$: Dynamic resource growth rate.
- $F_i(q_i, x_i)$: Resource consumption rate by robots allocated to area $i$.
- $x_i(t) \in [0, 1]$: Fraction of total robots allocated to area $i$ ($\sum_i x_i = 1$).

### 2. Consumption Model (Equation 3)
$$F_i(q_i, x_i) = R_i \cdot \tanh\left(\frac{\alpha_i q_i}{2}\right) \cdot x_i^{\beta_i}$$

Parameters fitted from robotic experiments:
- $R_i = 3.44$: Maximum consumption capacity.
- $\alpha_i = 0.036$: Saturation coefficient.
- $\beta_i = 0.91$: Interference / congestion penalty coefficient.

### 3. Payoff Mechanism (Equation 6)
$$p_i(t) = q_i(t) + \nu \left( -F_i(\gamma^*, x_i) + w_i \right)$$

- $\nu = 0$: **Model-free (reactive)** — payoff equals current backlog $q_i(t)$.
- $\nu > 0$: **Model-based (predictive)** — incorporates future net growth rate projected at maximum workload $\gamma^* = \max_j q_j$.

### 4. Revision Protocol (Equation 7)
When a robot's Poisson revision clock triggers (rate $\lambda = \frac{1}{8}\,\text{Hz}$):
- **Switching probability from area $i$ to area $j$**:
  $$\mathbb{P}(i \to j) = \rho [p_j - p_i]_+$$
- **Probability of remaining at area $i$**:
  $$\mathbb{P}(i \to i) = 1 - \rho \sum_{j \ne i} [p_j - p_i]_+$$

Where $[z]_+ = \max(0, z)$ and $\rho = \frac{1}{600}$.

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- Webots R2023b or later (Debian/Ubuntu package, Snap, or macOS bundle)

```bash
# Clone the repository
git clone https://github.com/VishaleeshwarR/Multi-robot-task-allocation.git
cd Multi-robot-task-allocation

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies and local package
pip install -r requirements.txt
pip install -e .
```

---

## Usage Guide

### 1. Webots 3D Interactive Simulation
Launch with the cross-platform runner `run.py`:

```bash
# Experiment 1: Temporary Workload Surge (Area 1)
python run.py -e 1

# Accelerated Quick Demo (100 seconds total, ideal for live demonstrations)
python run.py -e 1 -q

# Experiment 2: Permanent Growth Increase
python run.py -e 2 -q

# Experiment 3: 15 Robot Failures & Fleet Recovery
python run.py -e 3 -q

# Experiment 4: Model-Based Payoff Evaluation (nu=40)
python run.py -e 4 --nu 40 -q
```

### 2. Fast Headless Batch Simulation
To collect telemetry and automatically generate plots without opening the 3D GUI:

```bash
python run.py -e 1 -b -q
```
Upon completion, the supervisor exits automatically and writes:
- `results/webots_exp1_data.csv`
- `results/webots_exp1_measured.png`

### 3. Analytical Benchmark Solver
Run the pure ODE simulation to evaluate theoretical convergence:

```bash
python experiments/run_experiments.py
```
This generates comparison figures matching the paper's theoretical curves in `results/`.

---

## Project Structure

```
MultiRobotTaskAlloc/
├── controllers/
│   ├── supervisor_controller/
│   │   └── supervisor_controller.py   # Spawning, telemetry, proximity collection, experiment FSM
│   └── robot_controller/
│       └── robot_controller.py        # Poisson revision, RF communications, navigation, avoidance
├── models/
│   ├── consumption_model.py           # Resource dynamics & consumption (Eq. 2, 3)
│   ├── payoff_mechanism.py            # Model-free vs model-based payoff (Eq. 6)
│   └── revision_protocol.py           # Probabilistic task switching (Eq. 7)
├── experiments/
│   └── run_experiments.py             # Numerical ODE solver & analytical benchmarks
├── shared/
│   ├── communication.py               # Emitter/Receiver protocols & data serialization
│   ├── config.py                      # Global parameters and hyperparameter configurations
│   └── constants.py                   # Enumerations and system constants
├── results/                           # Telemetry CSVs and generated figures
├── worlds/
│   └── trash_collection.wbt          # 40 e-puck robot arena, 4 task zones, optimized physics
├── run.py                             # Cross-platform simulation CLI launcher
├── requirements.txt                   # Project dependencies
└── README.md                          # Documentation and project overview
```

---

## Citation

If you use this work in your research, please cite the original paper:

```bibtex
@inproceedings{park2021multi,
  title={Multi-Robot Task Allocation Games in Dynamically Changing Environments},
  author={Park, Shinkyu and Zhong, Yaofeng Desmond and Leonard, Naomi Ehrich},
  booktitle={2021 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={2501--2507},
  year={2021},
  organization={IEEE},
  doi={10.1109/ICRA48506.2021.9561849}
}
```

---

## License

This project is licensed under the MIT License.
