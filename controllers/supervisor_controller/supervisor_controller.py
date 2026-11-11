"""
Supervisor Controller for Multi-Robot Trash Collection
Based on Park, Zhong, Leonard (ICRA 2021)

Manages:
- Experiment scenario execution (Experiments 1, 2, 3, 4)
- Poisson trash spawning and dynamic growth rates
- System state tracking (q and x)
- Physical proximity trash collection
- Communication broadcast to all 40 robots
- Live telemetry logging and result plot generation
"""

from controller import Supervisor
import numpy as np
import random
import math
import struct
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Simulation parameters
TIME_STEP = 32
NUM_ROBOTS = 40
NUM_PATCHES = 4

# Patch parameters (from Example 1)
R = [3.44, 3.44, 3.44, 3.44]
ALPHA = [0.036, 0.036, 0.036, 0.036]
BETA = [0.91, 0.91, 0.91, 0.91]

# Patch boundaries
PATCH_BOUNDS = [
    (-6.75, -0.75, 0.75, 6.75),   # Patch 1: (x_min, x_max, y_min, y_max)
    (-6.75, -0.75, -6.75, -0.75), # Patch 2
    (0.75, 6.75, -6.75, -0.75),   # Patch 3
    (0.75, 6.75, 0.75, 6.75)      # Patch 4
]

PATCH_LABELS = ["Area 1", "Area 2", "Area 3", "Area 4"]
PATCH_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
TRASH_SIZE = 0.05

class SupervisorController:
    def __init__(self):
        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())
        
        # Communication
        self.emitter = self.supervisor.getDevice('emitter')
        self.emitter.setChannel(1)
        self.receiver = self.supervisor.getDevice('receiver')
        self.receiver.setChannel(2)
        self.receiver.enable(self.timestep)
        
        # Read experiment configuration from environment
        self.experiment = int(os.environ.get("EXPERIMENT", "1"))
        self.quick_mode = os.environ.get("EXP_QUICK", "0") == "1"
        self.headless = os.environ.get("HEADLESS", "0") == "1"
        self.nu = float(os.environ.get("NU", "0"))
        
        # Dynamic growth rates and active robots
        self.W = [0.5, 0.5, 0.5, 0.5]
        self.active_robots = NUM_ROBOTS
        
        # Set experiment timeline parameters
        if self.experiment == 1:
            self.surge_start = 20.0 if self.quick_mode else 500.0
            self.surge_end = 60.0 if self.quick_mode else 600.0
            self.duration = 100.0 if self.quick_mode else 1000.0
            self.exp_name = "Exp 1: Temporary Surge in Area 1"
        elif self.experiment == 2:
            self.surge_start = 20.0 if self.quick_mode else 500.0
            self.surge_end = float('inf')
            self.duration = 100.0 if self.quick_mode else 2000.0
            self.exp_name = "Exp 2: Permanent Growth Increase in Area 1"
        elif self.experiment == 3:
            self.fail_start = 20.0 if self.quick_mode else 500.0
            self.fail_end = 60.0 if self.quick_mode else 1500.0
            self.duration = 100.0 if self.quick_mode else 2000.0
            self.exp_name = "Exp 3: 15 Robot Failures"
        elif self.experiment == 4:
            self.W = [1.5, 0.5, 0.5, 0.5]
            if "NU" not in os.environ:
                self.nu = 40.0
            self.duration = 100.0 if self.quick_mode else 1000.0
            self.exp_name = f"Exp 4: Payoff Parameter nu={self.nu}"
        else:
            self.duration = 1000.0
            self.exp_name = "Default Baseline"

        # State variables
        self.q = [0.0] * NUM_PATCHES
        self.x = [0.25] * NUM_PATCHES
        robots_per_patch = max(1, NUM_ROBOTS // NUM_PATCHES)
        self.robot_assignments = [min(i // robots_per_patch, NUM_PATCHES - 1) for i in range(NUM_ROBOTS)]
        
        # Trash objects
        self.trash_objects = [[] for _ in range(NUM_PATCHES)]
        self.trash_id_counter = 0
        
        # Robot nodes
        self.robot_nodes = [self.supervisor.getFromDef(f"EPUCK_{i}") for i in range(NUM_ROBOTS)]
        
        # Poisson process tracking
        self.last_spawn_time = [0.0] * NUM_PATCHES
        self.next_spawn_time = [self.sample_spawn_time(i) for i in range(NUM_PATCHES)]
        
        # Telemetry recording
        self.time = 0.0
        self.stats_update_interval = 1000  # ms
        self.last_stats_time = 0.0
        self.history = []
        
        # Results directory
        self.results_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        print("="*65)
        print(f"WEBOTS SUPERVISOR INITIALIZED — {self.exp_name.upper()}")
        print(f"  Experiment:   {self.experiment}")
        print(f"  Mode:         {'Quick Demo (Accelerated)' if self.quick_mode else 'Standard Paper Timeline'}")
        print(f"  Target time:  {self.duration:.1f} s")
        print(f"  Initial W:    {self.W}")
        print(f"  nu value:     {self.nu}")
        print("="*65, flush=True)

    def sample_spawn_time(self, patch_idx):
        """Sample next trash spawn time from exponential distribution"""
        rate = self.W[patch_idx]
        if rate <= 0:
            return float('inf')
        return random.expovariate(rate) * 1000  # ms

    def spawn_trash(self, patch_idx):
        """Spawn a visual trash box in the specified patch"""
        # Limit active visual markers to 25 per patch to keep Webots simulation fast
        if len(self.trash_objects[patch_idx]) >= 25:
            self.q[patch_idx] += 1.0
            return

        bounds = PATCH_BOUNDS[patch_idx]
        x_min, x_max, y_min, y_max = bounds
        margin = 0.6
        x = random.uniform(x_min + margin, x_max - margin)
        y = random.uniform(y_min + margin, y_max - margin)
        z = TRASH_SIZE / 2
        
        trash_name = f"trash_{self.trash_id_counter}"
        self.trash_id_counter += 1
        
        trash_string = f"""
DEF {trash_name} Solid {{
  translation {x:.3f} {y:.3f} {z:.3f}
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor 1 0 0
        metalness 0
        roughness 0.5
      }}
      geometry Box {{
        size {TRASH_SIZE} {TRASH_SIZE} {TRASH_SIZE}
      }}
    }}
  ]
  name "{trash_name}"
}}
"""
        root = self.supervisor.getRoot()
        children_field = root.getField('children')
        children_field.importMFNodeFromString(-1, trash_string)
        
        trash_node = self.supervisor.getFromDef(trash_name)
        if trash_node:
            self.trash_objects[patch_idx].append(trash_node)
            self.q[patch_idx] = float(len(self.trash_objects[patch_idx]))

    def check_trash_spawning(self):
        """Check and spawn trash based on Poisson process"""
        current_time_ms = self.time * 1000
        for i in range(NUM_PATCHES):
            if current_time_ms >= self.last_spawn_time[i] + self.next_spawn_time[i]:
                self.spawn_trash(i)
                self.last_spawn_time[i] = current_time_ms
                self.next_spawn_time[i] = self.sample_spawn_time(i)

    def receive_robot_status(self):
        """Receive status updates from robots"""
        while self.receiver.getQueueLength() > 0:
            floats = self.receiver.getFloats()
            if len(floats) >= 2:
                robot_id = int(floats[0])
                current_patch = int(floats[1])
                if 0 <= robot_id < NUM_ROBOTS:
                    self.robot_assignments[robot_id] = current_patch
            self.receiver.nextPacket()

    def update_distribution(self):
        """Update robot distribution x based on active robot assignments"""
        counts = [0] * NUM_PATCHES
        active_count = 0
        for r_id in range(self.active_robots):
            assignment = self.robot_assignments[r_id]
            if 0 <= assignment < NUM_PATCHES:
                counts[assignment] += 1
                active_count += 1
        
        divisor = max(1, active_count)
        self.x = [count / divisor for count in counts]

    def broadcast_state(self):
        """Broadcast state and active trash positions to all robots"""
        # Header (15 floats):
        # [0..3]: q, [4..7]: x, [8]: time, [9..12]: W, [13]: nu, [14]: active_robots
        message_floats = [
            float(self.q[0]), float(self.q[1]), float(self.q[2]), float(self.q[3]),
            float(self.x[0]), float(self.x[1]), float(self.x[2]), float(self.x[3]),
            float(self.time),
            float(self.W[0]), float(self.W[1]), float(self.W[2]), float(self.W[3]),
            float(self.nu),
            float(self.active_robots)
        ]
        
        # Append up to 4 trash coordinates per patch (32 floats)
        max_trash_per_patch = 4
        for patch_idx in range(NUM_PATCHES):
            coords = []
            for t_node in self.trash_objects[patch_idx][:max_trash_per_patch]:
                if t_node:
                    try:
                        pos = t_node.getPosition()
                        coords.extend([float(pos[0]), float(pos[1])])
                    except Exception:
                        pass
            while len(coords) < max_trash_per_patch * 2:
                coords.extend([999.0, 999.0])
            message_floats.extend(coords)
        
        self.emitter.send(message_floats)

    def check_proximity_pickup(self):
        """Check if any active robot is close enough to collect trash"""
        pickup_radius = 0.35
        for patch_idx in range(NUM_PATCHES):
            for trash_node in self.trash_objects[patch_idx][:]:
                if trash_node is None:
                    continue
                try:
                    t_pos = trash_node.getPosition()
                    tx, ty = t_pos[0], t_pos[1]
                except Exception:
                    self.trash_objects[patch_idx].remove(trash_node)
                    continue
                
                # Check proximity against active robots only
                collected = False
                for r_id in range(self.active_robots):
                    r_node = self.robot_nodes[r_id]
                    if r_node is None:
                        continue
                    try:
                        r_pos = r_node.getPosition()
                        rx, ry = r_pos[0], r_pos[1]
                        if math.hypot(rx - tx, ry - ty) < pickup_radius:
                            trash_node.remove()
                            self.trash_objects[patch_idx].remove(trash_node)
                            self.q[patch_idx] = float(len(self.trash_objects[patch_idx]))
                            collected = True
                            break
                    except Exception:
                        pass
                if collected:
                    continue

    def update_experiment_state(self):
        """Update dynamic experiment scenario conditions"""
        if self.experiment == 1:
            if self.surge_start <= self.time <= self.surge_end:
                if self.W[0] != 5.0:
                    self.W[0] = 5.0
                    print(f"\n⚡ [EXPERIMENT 1 EVENT] Workload surge began in Area 1 (W[0] = 5.0) at t={self.time:.1f}s!\n", flush=True)
            else:
                if self.W[0] != 0.5:
                    self.W[0] = 0.5
                    print(f"\n🔄 [EXPERIMENT 1 EVENT] Workload surge ended in Area 1 (W[0] = 0.5) at t={self.time:.1f}s!\n", flush=True)

        elif self.experiment == 2:
            if self.time >= self.surge_start:
                if self.W[0] != 5.0:
                    self.W[0] = 5.0
                    print(f"\n⚡ [EXPERIMENT 2 EVENT] Permanent growth increase began in Area 1 (W[0] = 5.0) at t={self.time:.1f}s!\n", flush=True)

        elif self.experiment == 3:
            if self.fail_start <= self.time <= self.fail_end:
                if self.active_robots != NUM_ROBOTS - 15:
                    self.active_robots = NUM_ROBOTS - 15
                    print(f"\n⚠️  [EXPERIMENT 3 EVENT] 15 robots failed! Active fleet: {self.active_robots} at t={self.time:.1f}s!\n", flush=True)
            else:
                if self.active_robots != NUM_ROBOTS:
                    self.active_robots = NUM_ROBOTS
                    print(f"\n✅ [EXPERIMENT 3 EVENT] All 15 robots recovered! Active fleet: {self.active_robots} at t={self.time:.1f}s!\n", flush=True)

    def record_telemetry(self):
        """Log state telemetry to history array"""
        entry = {
            "time": self.time,
            "q0": self.q[0], "q1": self.q[1], "q2": self.q[2], "q3": self.q[3],
            "x0": self.x[0], "x1": self.x[1], "x2": self.x[2], "x3": self.x[3],
            "w0": self.W[0], "w1": self.W[1], "w2": self.W[2], "w3": self.W[3],
            "active_robots": self.active_robots
        }
        self.history.append(entry)

    def print_statistics(self):
        """Display live status table in console"""
        print(f"[{self.exp_name}] t={self.time:5.1f}s | "
              f"Alloc: [{self.x[0]*100:4.1f}%, {self.x[1]*100:4.1f}%, {self.x[2]*100:4.1f}%, {self.x[3]*100:4.1f}%] | "
              f"q: [{self.q[0]:4.1f}, {self.q[1]:4.1f}, {self.q[2]:4.1f}, {self.q[3]:4.1f}] | "
              f"Active Bots: {self.active_robots}", flush=True)

    def save_results(self):
        """Save telemetry CSV and generate plots"""
        if not self.history:
            return

        csv_file = os.path.join(self.results_dir, f"webots_exp{self.experiment}_data.csv")
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.history[0].keys()))
            writer.writeheader()
            writer.writerows(self.history)
        print(f"\n✓ Saved Webots simulation telemetry: {csv_file}")

        # Generate plot from Webots data
        times = [e["time"] for e in self.history]
        q_data = np.array([[e[f"q{i}"] for e in self.history] for i in range(NUM_PATCHES)])
        x_data = np.array([[e[f"x{i}"] * 100 for e in self.history] for i in range(NUM_PATCHES)])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"Webots Simulation — {self.exp_name}", fontsize=13, fontweight="bold")

        for i in range(NUM_PATCHES):
            ax1.plot(times, q_data[i], label=PATCH_LABELS[i], color=PATCH_COLORS[i], lw=2.2)
            ax2.plot(times, x_data[i], label=PATCH_LABELS[i], color=PATCH_COLORS[i], lw=2.2)

        ax1.set_xlabel("Simulated Time (s)")
        ax1.set_ylabel("Measured Trash Objects q_i")
        ax1.set_title("Resource Levels per Area (Webots)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)

        ax2.set_xlabel("Simulated Time (s)")
        ax2.set_ylabel("Robot Allocation (%)")
        ax2.set_title("Robot Allocation per Area (Webots)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)

        plot_file = os.path.join(self.results_dir, f"webots_exp{self.experiment}_measured.png")
        fig.tight_layout()
        fig.savefig(plot_file, dpi=150)
        plt.close(fig)
        print(f"✓ Saved Webots simulation plot: {plot_file}\n")

    def run(self):
        """Main supervisor loop"""
        while self.supervisor.step(self.timestep) != -1:
            self.time = self.supervisor.getTime()
            
            # Update experiment phase
            self.update_experiment_state()
            
            # Receive robot status
            self.receive_robot_status()
            
            # Update robot distribution
            self.update_distribution()
            
            # Check for trash spawning
            self.check_trash_spawning()
            
            # Check for physical trash pickup by active robots
            self.check_proximity_pickup()
            
            # Broadcast state and trash positions to robots
            self.broadcast_state()
            
            # Record telemetry and print status every 1 second
            if (self.time * 1000) - self.last_stats_time >= self.stats_update_interval:
                self.record_telemetry()
                self.print_statistics()
                self.last_stats_time = self.time * 1000
            
            # Check completion
            if self.time >= self.duration:
                print("\n" + "="*65)
                print(f"EXPERIMENT SIMULATION COMPLETE (t = {self.time:.1f} s)")
                print("="*65)
                self.save_results()
                if self.headless:
                    self.supervisor.simulationQuit(0)
                break

if __name__ == "__main__":
    supervisor = SupervisorController()
    supervisor.run()
