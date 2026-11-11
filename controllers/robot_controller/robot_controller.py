"""
Multi-Robot Trash Collection Controller
Based on Park, Zhong, Leonard (ICRA 2021)

Implements:
- Game-theoretic task allocation
- Poisson clock revision protocol
- Payoff-based decision making
- Finite state machine for robot behavior
"""

from controller import Robot, GPS, Compass, DistanceSensor, Emitter, Receiver
import numpy as np
import random
import math
import struct

# Simulation parameters
TIME_STEP = 64  # ms
NUM_ROBOTS = 40
NUM_PATCHES = 4
LAMBDA = 0.125  # Poisson clock parameter (1/8 seconds)
RHO = 1/600     # Task revision protocol rate
NU = 0          # Payoff mechanism weight (0 for model-free)

# Patch parameters (from Example 1 in paper)
R = [3.44, 3.44, 3.44, 3.44]      # Maximum consumption rate
ALPHA = [0.036, 0.036, 0.036, 0.036]  # Saturation parameter
BETA = [0.91, 0.91, 0.91, 0.91]   # Power parameter
W = [0.5, 0.5, 0.5, 0.5]          # Growth rates

# Patch centers
PATCH_CENTERS = [
    (-3.75, 3.75),   # Patch 1 (top-left)
    (-3.75, -3.75),  # Patch 2 (bottom-left)
    (3.75, -3.75),   # Patch 3 (bottom-right)
    (3.75, 3.75)     # Patch 4 (top-right)
]

# Dumpster locations
DUMPSTERS = [
    (-6.5, 6.5),    # Dumpster 1
    (-6.5, -6.5),   # Dumpster 2
    (6.5, -6.5),    # Dumpster 3
    (6.5, 6.5)      # Dumpster 4
]

# Robot states
STATE_FORAGING = 0
STATE_VISUAL_SERVOING = 1
STATE_PICKUP = 2
STATE_COLLISION_AVOIDANCE = 3
STATE_TRANSITION = 4
STATE_DUMP = 5

class RobotController:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        # Get robot ID from name
        self.name = self.robot.getName()
        self.robot_id = int(self.name.split('_')[1])
        
        # Initialize devices
        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)
        
        self.compass = self.robot.getDevice('compass')
        self.compass.enable(self.timestep)
        
        # Initialize motors
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        
        # Initialize distance sensors
        self.distance_sensors = []
        for i in range(8):
            sensor = self.robot.getDevice(f'ps{i}')
            sensor.enable(self.timestep)
            self.distance_sensors.append(sensor)
        
        # Communication (channel 2: send to supervisor, channel 1: receive from supervisor)
        self.emitter = self.robot.getDevice('emitter')
        self.emitter.setChannel(2)
        self.receiver = self.robot.getDevice('receiver')
        self.receiver.setChannel(1)
        self.receiver.enable(self.timestep)
        
        # State variables
        self.state = STATE_FORAGING
        self.previous_state = STATE_FORAGING
        robots_per_patch = max(1, NUM_ROBOTS // NUM_PATCHES)
        self.current_patch = min(self.robot_id // robots_per_patch, NUM_PATCHES - 1)  # Initial assignment
        self.target_patch = self.current_patch
        self.trash_collected = 0
        self.max_capacity = 10
        
        # Game-theoretic variables
        self.q = [0.0] * NUM_PATCHES  # Trash volumes (received from supervisor)
        self.x = [0.25] * NUM_PATCHES  # Robot distribution
        self.W = [0.5, 0.5, 0.5, 0.5]  # Growth rates
        self.nu = 0.0                  # Payoff parameter
        self.active_robots = NUM_ROBOTS # Fleet health / active count
        self.patch_trash_coords = [[] for _ in range(NUM_PATCHES)]  # Active trash locations
        self.last_revision_time = 0
        self.next_revision_time = self.sample_poisson_time()
        
        # Navigation
        self.target_position = PATCH_CENTERS[self.current_patch]
        self.collision_avoidance_time = 0
        
        print(f"Robot {self.robot_id} initialized in patch {self.current_patch}")
    
    def sample_poisson_time(self):
        """Sample next revision time from exponential distribution"""
        return random.expovariate(LAMBDA) * 1000  # Convert to ms
    
    def consumption_rate(self, qi, xi, patch_idx):
        """Consumption rate F_i from Equation (3)"""
        exp_term = math.exp(ALPHA[patch_idx] * qi)
        saturating = (exp_term - 1) / (exp_term + 1)
        return R[patch_idx] * saturating * (xi ** BETA[patch_idx])
    
    def payoff(self, patch_idx):
        """Payoff mechanism from Equation (6)"""
        if self.nu == 0:
            # Model-free: payoff is just trash volume
            return self.q[patch_idx]
        else:
            # Model-based
            gamma_star = max(self.q)
            return (self.q[patch_idx] + 
                   self.nu * (-self.consumption_rate(gamma_star, self.x[patch_idx], patch_idx) + 
                         self.W[patch_idx]))
    
    def revision_protocol(self):
        """Task revision protocol from Equation (7)"""
        # Calculate payoffs for all patches
        payoffs = [self.payoff(i) for i in range(NUM_PATCHES)]
        
        # Current patch payoff
        current_payoff = payoffs[self.current_patch]
        
        # Scale RHO adaptively when trash volumes are low so allocation responds actively
        max_payoff = max(payoffs)
        scale = 1.0 if max_payoff > 50 else (10.0 if max_payoff > 10 else 50.0)
        effective_rho = RHO * scale
        
        # Calculate switching probabilities
        switch_probs = []
        for i in range(NUM_PATCHES):
            if i == self.current_patch:
                switch_probs.append(0.0)
            else:
                diff = payoffs[i] - current_payoff
                switch_probs.append(effective_rho * max(0.0, diff))
        
        total = sum(switch_probs)
        if total <= 0:
            return self.current_patch
        
        # Cap total switching probability at 0.8
        if total > 0.8:
            switch_probs = [p * 0.8 / total for p in switch_probs]
            total = 0.8
        
        # Make decision based on cumulative probabilities
        rand = random.random()
        cumulative = 0.0
        
        for i in range(NUM_PATCHES):
            if i != self.current_patch:
                cumulative += switch_probs[i]
                if rand < cumulative:
                    # Switch to patch i
                    print(f"Robot {self.robot_id}: Switching from patch {self.current_patch} to {i}", flush=True)
                    print(f"  Payoffs: {[f'{p:.2f}' for p in payoffs]}", flush=True)
                    return i
        
        # Stay in current patch
        return self.current_patch
    
    def receive_state_info(self):
        """Receive state, experiment parameters, and active trash positions from supervisor"""
        while self.receiver.getQueueLength() > 0:
            try:
                floats = self.receiver.getFloats()
                if len(floats) >= 9:
                    self.q = list(floats[0:4])
                    self.x = list(floats[4:8])
                if len(floats) >= 15:
                    self.W = list(floats[9:13])
                    self.nu = floats[13]
                    self.active_robots = int(floats[14])
                if len(floats) >= 47:
                    max_trash_per_patch = 4
                    coords_per_patch = []
                    for p in range(NUM_PATCHES):
                        p_trash = []
                        offset = 15 + p * (max_trash_per_patch * 2)
                        for k in range(max_trash_per_patch):
                            tx = floats[offset + 2 * k]
                            ty = floats[offset + 2 * k + 1]
                            if tx < 500.0:  # Not padded with 999.0
                                p_trash.append((tx, ty))
                        coords_per_patch.append(p_trash)
                    self.patch_trash_coords = coords_per_patch
            except Exception as e:
                pass
            
            self.receiver.nextPacket()
    
    def get_current_patch_trash(self):
        """Get active trash coordinates in current patch"""
        if hasattr(self, 'patch_trash_coords') and len(self.patch_trash_coords) > self.current_patch:
            return self.patch_trash_coords[self.current_patch]
        return []
    
    def send_status(self):
        """Send robot status to supervisor using native floats"""
        # Pack: robot_id, current_patch, trash_collected (all converted to floats)
        message_floats = [
            float(self.robot_id), 
            float(self.current_patch), 
            float(self.trash_collected)
        ]
        self.emitter.send(message_floats)
    def get_position(self):
        """Get current robot position"""
        pos = self.gps.getValues()
        return (pos[0], pos[1])
    
    def get_heading(self):
        """Get current heading angle"""
        north = self.compass.getValues()
        heading = math.atan2(north[0], north[1])
        return heading
    
    def distance_to(self, target):
        """Calculate distance to target position"""
        pos = self.get_position()
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        return math.sqrt(dx**2 + dy**2)
    
    def angle_to(self, target):
        """Calculate angle to target position"""
        pos = self.get_position()
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        target_angle = math.atan2(dy, dx)
        heading = self.get_heading()
        angle_diff = target_angle - heading
        
        # Normalize to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        return angle_diff
    
    def detect_collision(self):
        """Check for obstacles using front distance sensors"""
        threshold = 120  # Front sensor threshold
        # E-puck distance sensors: ps0 (front right), ps1 (front right 45°), ps6 (front left 45°), ps7 (front left)
        front_sensors = [self.distance_sensors[0], self.distance_sensors[1],
                         self.distance_sensors[6], self.distance_sensors[7]]
        for sensor in front_sensors:
            if sensor.getValue() > threshold:
                return True
        return False
    
    def move_to_target(self, target, max_speed=6.28):
        """Move towards target with orientation control"""
        angle_diff = self.angle_to(target)
        
        # If not aligned with target, pivot in place first
        if abs(angle_diff) > 0.5:
            turn_speed = 3.5
            if angle_diff > 0:
                # Target is to our left -> turn left (CCW)
                left_speed = -turn_speed
                right_speed = turn_speed
            else:
                # Target is to our right -> turn right (CW)
                left_speed = turn_speed
                right_speed = -turn_speed
        else:
            # Aligned: drive forward with proportional heading correction
            angular_speed = 2.5 * angle_diff
            left_speed = max_speed - angular_speed
            right_speed = max_speed + angular_speed
            
            # Clamp speeds
            left_speed = max(-max_speed, min(max_speed, left_speed))
            right_speed = max(-max_speed, min(max_speed, right_speed))
        
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)
    
    def stop(self):
        """Stop the robot"""
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
    
    def collision_avoidance(self):
        """Collision avoidance behavior using distance sensors"""
        sensors = [s.getValue() for s in self.distance_sensors]
        
        # E-puck distance sensors:
        # ps0 (front right 10°), ps1 (front right 45°), ps2 (right 90°)
        # ps5 (left 90°), ps6 (front left 45°), ps7 (front left 10°)
        right_obstacle = sensors[0] + sensors[1] + sensors[2]
        left_obstacle = sensors[5] + sensors[6] + sensors[7]
        
        # Avoid obstacles with decisive steering
        if left_obstacle > 150 and right_obstacle > 150:
            # Front obstacle: turn right
            left_speed = 3.0
            right_speed = -3.0
        elif left_obstacle > right_obstacle:
            # Obstacle on left: turn right
            left_speed = 3.0
            right_speed = -2.0
        elif right_obstacle > left_obstacle:
            # Obstacle on right: turn left
            left_speed = -2.0
            right_speed = 3.0
        else:
            left_speed = 3.0
            right_speed = -3.0
        
        # Clamp speeds to motor limits (-6.28 to 6.28 rad/s)
        left_speed = max(-6.28, min(6.28, left_speed))
        right_speed = max(-6.28, min(6.28, right_speed))
        
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)
    
    def check_revision_time(self):
        """Check if it's time to revise task selection (Poisson clock)"""
        current_time = self.robot.getTime() * 1000  # Convert to ms
        
        if current_time >= self.last_revision_time + self.next_revision_time:
            # Only revise task if currently foraging (don't interrupt transition or dumping)
            if self.state == STATE_FORAGING:
                new_patch = self.revision_protocol()
                
                if new_patch != self.current_patch:
                    self.target_patch = new_patch
                    self.state = STATE_TRANSITION
            
            # Sample next revision time
            self.last_revision_time = current_time
            self.next_revision_time = self.sample_poisson_time()
    
    def run(self):
        """Main control loop"""
        step_count = 0
        while self.robot.step(self.timestep) != -1:
            step_count += 1
            
            # Receive state information from supervisor
            self.receive_state_info()
            
            # Send status to supervisor
            self.send_status()
            
            # If deactivated (e.g. during robot failure experiment), stop motors and idle
            if self.robot_id >= self.active_robots:
                self.stop()
                continue
            
            # Check for task revision (Poisson clock)
            self.check_revision_time()
            
            # Finite state machine
            if self.state == STATE_FORAGING:
                patch_trash = self.get_current_patch_trash()
                
                # If trash exists in current patch, seek the nearest one
                if len(patch_trash) > 0:
                    closest_trash = min(patch_trash, key=lambda t: self.distance_to(t))
                    self.target_position = closest_trash
                    target_speed = 4.5
                    
                    # Track individual pickup when reaching trash
                    if self.distance_to(closest_trash) < 0.35:
                        self.trash_collected += 1
                else:
                    # No trash in current patch: patrol inside the patch bounds
                    patch_center = PATCH_CENTERS[self.current_patch]
                    if self.distance_to(self.target_position) < 0.6 or random.random() < 0.02:
                        ox = random.uniform(-2.2, 2.2)
                        oy = random.uniform(-2.2, 2.2)
                        self.target_position = (patch_center[0] + ox, patch_center[1] + oy)
                    target_speed = 3.5
                
                # Check for collisions with other robots or arena walls
                if self.detect_collision():
                    self.previous_state = self.state
                    self.state = STATE_COLLISION_AVOIDANCE
                    self.collision_avoidance_time = self.robot.getTime()
                else:
                    self.move_to_target(self.target_position, max_speed=target_speed)
            
            elif self.state == STATE_COLLISION_AVOIDANCE:
                self.collision_avoidance()
                
                # Return to previous state after 0.5 seconds
                if self.robot.getTime() - self.collision_avoidance_time > 0.5:
                    self.state = self.previous_state
            
            elif self.state == STATE_TRANSITION:
                # Transition across the arena to new patch
                target_patch_center = PATCH_CENTERS[self.target_patch]
                
                # Check if arrived at target patch (within 2.0m of patch center)
                if self.distance_to(target_patch_center) < 2.0:
                    self.current_patch = self.target_patch
                    self.target_position = target_patch_center
                    self.state = STATE_FORAGING
                    print(f"Robot {self.robot_id}: Arrived at patch {self.current_patch}", flush=True)
                # Collision avoidance during transition
                elif self.detect_collision():
                    self.previous_state = self.state
                    self.state = STATE_COLLISION_AVOIDANCE
                    self.collision_avoidance_time = self.robot.getTime()
                else:
                    self.move_to_target(target_patch_center, max_speed=5.5)

if __name__ == "__main__":
    controller = RobotController()
    controller.run()
