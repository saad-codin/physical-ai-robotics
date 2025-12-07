---
sidebar_position: 5
---

# Mobile Robot Navigation

Mobile robot navigation is the process of moving a robot from one location to another while avoiding obstacles and following optimal paths. This section covers the fundamental concepts, algorithms, and techniques for autonomous robot navigation.

## Navigation Stack Overview

### The Navigation Stack

Modern robot navigation typically follows a hierarchical approach with multiple layers:

```
High-level Planning → Global Path Planning → Local Path Planning → Motion Control
```

Each layer operates at different time scales and spatial resolutions:

- **High-level planning**: Task planning and mission management
- **Global path planning**: Route finding on static maps
- **Local path planning**: Obstacle avoidance in dynamic environments
- **Motion control**: Low-level motor commands

## Mapping and Localization

### Simultaneous Localization and Mapping (SLAM)

SLAM is crucial for navigation in unknown environments:

```python
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal

class ParticleFilterSLAM:
    def __init__(self, num_particles=1000):
        self.num_particles = num_particles
        self.particles = self.initialize_particles()
        self.map = OccupancyGrid()
        self.sensor_model = SensorModel()
        self.motion_model = MotionModel()

    def initialize_particles(self):
        """Initialize particles with random poses"""
        particles = []
        for _ in range(self.num_particles):
            particle = {
                'pose': np.random.uniform(-10, 10, 3),  # x, y, theta
                'weight': 1.0 / self.num_particles,
                'map': OccupancyGrid()  # Local map for each particle
            }
            particles.append(particle)
        return particles

    def predict(self, control_input, dt):
        """Predict particle poses based on motion model"""
        for particle in self.particles:
            # Sample from motion model
            new_pose = self.motion_model.sample(
                particle['pose'], control_input, dt
            )
            particle['pose'] = new_pose

    def update(self, sensor_data):
        """Update particle weights based on sensor observations"""
        total_weight = 0

        for particle in self.particles:
            # Calculate likelihood of observation given particle pose
            likelihood = self.sensor_model.likelihood(
                sensor_data, particle['pose'], particle['map']
            )
            particle['weight'] *= likelihood
            total_weight += particle['weight']

        # Normalize weights
        if total_weight > 0:
            for particle in self.particles:
                particle['weight'] /= total_weight

        # Resample particles
        self.resample()

    def resample(self):
        """Resample particles based on weights"""
        # Systematic resampling
        weights = [p['weight'] for p in self.particles]
        cumulative_weights = np.cumsum(weights)

        new_particles = []
        step = 1.0 / self.num_particles
        start = np.random.uniform(0, step)

        for i in range(self.num_particles):
            index = np.searchsorted(cumulative_weights, start + i * step)
            new_particles.append(self.particles[index].copy())

        self.particles = new_particles

    def estimate_pose(self):
        """Estimate current pose from particles"""
        weighted_poses = []
        weights = []

        for particle in self.particles:
            weighted_poses.append(particle['pose'] * particle['weight'])
            weights.append(particle['weight'])

        estimated_pose = np.sum(weighted_poses, axis=0) / np.sum(weights)
        return estimated_pose
```

### Occupancy Grid Mapping

Occupancy grid maps represent the environment as a grid of occupied/empty probabilities:

```python
class OccupancyGrid:
    def __init__(self, width=100, height=100, resolution=0.1):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin = np.array([0, 0])

        # Initialize with unknown probabilities (0.5)
        self.grid = np.full((height, width), 0.5, dtype=np.float32)

    def world_to_grid(self, world_x, world_y):
        """Convert world coordinates to grid coordinates"""
        grid_x = int((world_x - self.origin[0]) / self.resolution)
        grid_y = int((world_y - self.origin[1]) / self.resolution)
        return grid_x, grid_y

    def grid_to_world(self, grid_x, grid_y):
        """Convert grid coordinates to world coordinates"""
        world_x = grid_x * self.resolution + self.origin[0]
        world_y = grid_y * self.resolution + self.origin[1]
        return world_x, world_y

    def update_ray(self, start, end, occupied_threshold=0.7):
        """Update grid using Bresenham's line algorithm for ray tracing"""
        start_grid = self.world_to_grid(start[0], start[1])
        end_grid = self.world_to_grid(end[0], end[1])

        # Bresenham's line algorithm
        points = self.bresenham_line(start_grid[0], start_grid[1],
                                   end_grid[0], end_grid[1])

        # Update each point along the ray
        for x, y in points[:-1]:  # All points except the last (occupied)
            if 0 <= x < self.width and 0 <= y < self.height:
                # Free space (decrease occupancy)
                self.grid[y, x] = self.log_odds_to_probability(
                    self.probability_to_log_odds(self.grid[y, x]) - 0.4
                )

        # Update endpoint (occupied)
        end_x, end_y = end_grid
        if 0 <= end_x < self.width and 0 <= end_y < self.height:
            self.grid[end_y, end_x] = self.log_odds_to_probability(
                self.probability_to_log_odds(self.grid[end_y, end_x]) + 0.6
            )

    def bresenham_line(self, x0, y0, x1, y1):
        """Bresenham's line algorithm"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x_step = 1 if x0 < x1 else -1
        y_step = 1 if y0 < y1 else -1
        error = dx - dy

        x, y = x0, y0
        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break
            error2 = 2 * error
            if error2 > -dy:
                error -= dy
                x += x_step
            if error2 < dx:
                error += dx
                y += y_step
        return points

    def probability_to_log_odds(self, prob):
        """Convert probability to log-odds"""
        prob = np.clip(prob, 0.001, 0.999)  # Avoid log(0)
        return np.log(prob / (1 - prob))

    def log_odds_to_probability(self, log_odds):
        """Convert log-odds to probability"""
        prob = 1 - 1 / (1 + np.exp(log_odds))
        return prob
```

## Global Path Planning

### A* Path Planning Algorithm

A* is a popular algorithm for finding optimal paths in static environments:

```python
import heapq
from collections import defaultdict

class AStarPlanner:
    def __init__(self, occupancy_grid):
        self.grid = occupancy_grid
        self.grid_data = occupancy_grid.grid

    def plan_path(self, start, goal, obstacle_threshold=0.7):
        """Plan path using A* algorithm"""
        start_grid = self.grid.world_to_grid(start[0], start[1])
        goal_grid = self.grid.world_to_grid(goal[0], goal[1])

        # Check if start and goal are valid
        if not self.is_valid_cell(start_grid[0], start_grid[1], obstacle_threshold):
            raise ValueError("Start position is in obstacle")
        if not self.is_valid_cell(goal_grid[0], goal_grid[1], obstacle_threshold):
            raise ValueError("Goal position is in obstacle")

        # Initialize data structures
        open_set = [(0, start_grid)]  # (f_score, (x, y))
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}

        open_set_hash = {start_grid}

        while open_set:
            current = heapq.heappop(open_set)[1]
            open_set_hash.remove(current)

            if current == goal_grid:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(current):
                if not self.is_valid_cell(neighbor[0], neighbor[1], obstacle_threshold):
                    continue

                tentative_g_score = g_score[current] + self.distance(current, neighbor)

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_grid)

                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)

        return None  # No path found

    def heuristic(self, pos1, pos2):
        """Heuristic function (Euclidean distance)"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def distance(self, pos1, pos2):
        """Distance between adjacent cells"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def get_neighbors(self, pos):
        """Get 8-connected neighbors"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor = (pos[0] + dx, pos[1] + dy)
                if (0 <= neighbor[0] < self.grid.width and
                    0 <= neighbor[1] < self.grid.height):
                    neighbors.append(neighbor)
        return neighbors

    def is_valid_cell(self, x, y, threshold):
        """Check if cell is valid (not occupied)"""
        if 0 <= x < self.grid.width and 0 <= y < self.grid.height:
            return self.grid_data[y, x] < threshold
        return False

    def reconstruct_path(self, came_from, current):
        """Reconstruct path from came_from dictionary"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()

        # Convert back to world coordinates
        world_path = []
        for grid_pos in path:
            world_pos = self.grid.grid_to_world(grid_pos[0], grid_pos[1])
            world_path.append(world_pos)

        return world_path
```

### Dijkstra's Algorithm

Dijkstra's algorithm finds shortest paths without heuristics:

```python
def dijkstra_pathfinding(grid, start, goal, obstacle_threshold=0.7):
    """
    Dijkstra's algorithm for path planning
    """
    height, width = grid.shape
    distances = np.full((height, width), np.inf)
    previous = np.full((height, width, 2), -1)

    start_x, start_y = start
    goal_x, goal_y = goal

    distances[start_y, start_x] = 0
    pq = [(0, (start_y, start_x))]

    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    costs = [np.sqrt(2), 1, np.sqrt(2), 1, 1, np.sqrt(2), 1, np.sqrt(2)]

    while pq:
        current_dist, (cur_y, cur_x) = heapq.heappop(pq)

        if (cur_y, cur_x) == (goal_y, goal_x):
            break

        if current_dist > distances[cur_y, cur_x]:
            continue

        for i, (dy, dx) in enumerate(directions):
            new_y, new_x = cur_y + dy, cur_x + dx

            if (0 <= new_y < height and 0 <= new_x < width and
                grid[new_y, new_x] < obstacle_threshold):

                new_dist = current_dist + costs[i]

                if new_dist < distances[new_y, new_x]:
                    distances[new_y, new_x] = new_dist
                    previous[new_y, new_x] = [cur_y, cur_x]
                    heapq.heappush(pq, (new_dist, (new_y, new_x)))

    # Reconstruct path
    path = []
    cur_y, cur_x = goal_y, goal_x

    while not (cur_y == start_y and cur_x == start_x):
        path.append((cur_x, cur_y))
        prev_y, prev_x = previous[cur_y, cur_x]

        if prev_y == -1 and prev_x == -1:  # No path found
            return None

        cur_y, cur_x = int(prev_y), int(prev_x)

    path.append((start_x, start_y))
    path.reverse()

    return path
```

## Local Path Planning and Obstacle Avoidance

### Dynamic Window Approach (DWA)

DWA is a local planning algorithm that considers robot dynamics:

```python
class DynamicWindowApproach:
    def __init__(self, robot_config):
        self.max_speed = robot_config['max_speed']
        self.min_speed = robot_config['min_speed']
        self.max_yawrate = robot_config['max_yawrate']
        self.max_accel = robot_config['max_accel']
        self.max_dyawrate = robot_config['max_dyawrate']
        self.v_resolution = robot_config['v_resolution']
        self.yawrate_resolution = robot_config['yawrate_resolution']
        self.dt = robot_config['dt']
        self.predict_time = robot_config['predict_time']
        self.to_goal_cost_gain = robot_config['to_goal_cost_gain']
        self.speed_cost_gain = robot_config['speed_cost_gain']
        self.obstacle_cost_gain = robot_config['obstacle_cost_gain']

    def plan_local_path(self, state, goal, obstacles):
        """
        Plan local path using Dynamic Window Approach
        """
        # Generate dynamic window
        dwa = self.calc_dynamic_window(state)

        # Evaluate trajectories
        best_traj = None
        min_cost = float('inf')

        for v in np.arange(dwa[0], dwa[1], self.v_resolution):
            for yawrate in np.arange(dwa[2], dwa[3], self.yawrate_resolution):
                # Simulate trajectory
                traj = self.predict_trajectory(state, v, yawrate)

                # Calculate costs
                to_goal_cost = self.calc_to_goal_cost(traj, goal)
                speed_cost = self.calc_speed_cost(traj)
                ob_cost = self.calc_obstacle_cost(traj, obstacles)

                final_cost = (
                    self.to_goal_cost_gain * to_goal_cost +
                    self.speed_cost_gain * speed_cost +
                    self.obstacle_cost_gain * ob_cost
                )

                if final_cost < min_cost:
                    min_cost = final_cost
                    best_traj = traj

        return best_traj

    def calc_dynamic_window(self, state):
        """
        Calculate dynamic window
        [v_min, v_max, yawrate_min, yawrate_max]
        """
        # Dynamic window from motion model
        vs = [self.min_speed, self.max_speed,
              -self.max_yawrate, self.max_yawrate]

        # Dynamic window from kinematic constraints
        vd = [
            state[3] - self.max_accel * self.dt,
            state[3] + self.max_accel * self.dt,
            state[4] - self.max_dyawrate * self.dt,
            state[4] + self.max_dyawrate * self.dt
        ]

        # Minimum of both
        dw = [max(vs[0], vd[0]), min(vs[1], vd[1]),
              max(vs[2], vd[2]), min(vs[3], vd[3])]

        return dw

    def predict_trajectory(self, state, v, yawrate):
        """
        Predict trajectory for given velocity and yawrate
        """
        x, y, theta, v, omega = state
        traj = np.array([x, y, theta, v, omega]).reshape(1, -1)

        time = 0
        while time <= self.predict_time:
            x, y, theta, v, omega = self.motion_model([x, y, theta, v, omega], [v, yawrate])
            traj = np.vstack((traj, [x, y, theta, v, omega]))
            time += self.dt

        return traj

    def motion_model(self, state, control):
        """
        Motion model for differential drive robot
        """
        x, y, theta, v, omega = state
        v_cmd, omega_cmd = control

        x_new = x + v * np.cos(theta) * self.dt
        y_new = y + v * np.sin(theta) * self.dt
        theta_new = theta + omega * self.dt
        v_new = v_cmd
        omega_new = omega_cmd

        return [x_new, y_new, theta_new, v_new, omega_new]

    def calc_to_goal_cost(self, traj, goal):
        """
        Calculate cost to goal
        """
        dx = goal[0] - traj[-1, 0]
        dy = goal[1] - traj[-1, 1]
        error_angle = np.arctan2(dy, dx)
        cost_angle = error_angle - traj[-1, 2]
        cost = abs(np.arctan2(np.sin(cost_angle), np.cos(cost_angle)))

        return cost

    def calc_speed_cost(self, traj):
        """
        Calculate speed cost
        """
        return abs(self.max_speed - traj[-1, 3])

    def calc_obstacle_cost(self, traj, obstacles):
        """
        Calculate obstacle cost
        """
        min_dist = float('inf')
        for point in traj:
            for obs in obstacles:
                dist = np.sqrt((point[0] - obs[0])**2 + (point[1] - obs[1])**2)
                if dist < min_dist:
                    min_dist = dist

        # Cost is inverse of distance
        return 1.0 / min_dist if min_dist != 0 else float('inf')
```

### Vector Field Histogram (VFH)

VFH is another local navigation approach:

```python
class VectorFieldHistogram:
    def __init__(self, robot_radius=0.3, sensor_range=3.0, num_sectors=72):
        self.robot_radius = robot_radius
        self.sensor_range = sensor_range
        self.num_sectors = num_sectors
        self.sector_width = 2 * np.pi / num_sectors

    def plan_with_vfh(self, current_pose, goal_pose, sensor_data):
        """
        Plan direction using Vector Field Histogram
        """
        # Create polar histogram from sensor data
        histogram = self.create_histogram(sensor_data)

        # Calculate target direction to goal
        target_direction = self.calculate_target_direction(current_pose, goal_pose)

        # Find valid directions
        valid_directions = self.find_valid_directions(histogram)

        # Select best direction
        best_direction = self.select_best_direction(
            valid_directions, target_direction, current_pose[2]
        )

        return best_direction

    def create_histogram(self, sensor_data):
        """
        Create polar histogram from sensor readings
        """
        histogram = np.zeros(self.num_sectors)

        # Process each sensor reading
        for angle, distance in enumerate(sensor_data):
            if distance < self.sensor_range:
                # Calculate which sector this reading belongs to
                sector = int(angle / len(sensor_data) * self.num_sectors)
                histogram[sector] = max(histogram[sector], distance)

        # Convert to binary: 1 for free space, 0 for obstacles
        threshold = self.robot_radius * 2  # Consider robot size
        binary_histogram = (histogram > threshold).astype(int)

        return binary_histogram

    def find_valid_directions(self, histogram):
        """
        Find all valid (free) directions
        """
        valid_directions = []

        for i in range(len(histogram)):
            if histogram[i] == 1:  # Free space
                angle = i * self.sector_width - np.pi  # Convert to -π to π
                valid_directions.append(angle)

        return valid_directions

    def select_best_direction(self, valid_directions, target_direction, current_heading):
        """
        Select best direction considering target and current heading
        """
        if not valid_directions:
            return current_heading  # No valid directions, maintain current heading

        # Calculate cost for each valid direction
        best_direction = valid_directions[0]
        min_cost = float('inf')

        for direction in valid_directions:
            # Cost based on deviation from target direction
            target_cost = abs(self.normalize_angle(direction - target_direction))

            # Cost based on deviation from current heading (smooth turning)
            heading_cost = abs(self.normalize_angle(direction - current_heading)) * 0.5

            total_cost = target_cost + heading_cost

            if total_cost < min_cost:
                min_cost = total_cost
                best_direction = direction

        return best_direction

    def normalize_angle(self, angle):
        """
        Normalize angle to [-π, π] range
        """
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
```

## Navigation Architecture

### ROS 2 Navigation Stack Integration

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformListener, Buffer
import tf2_geometry_msgs

class NavigationNode(Node):
    def __init__(self):
        super().__init__('navigation_node')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/global_plan', 10)
        self.local_plan_pub = self.create_publisher(Path, '/local_plan', 10)

        # Subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.goal_sub = self.create_subscription(
            PoseStamped, '/move_base_simple/goal', self.goal_callback, 10
        )

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Navigation components
        self.global_planner = AStarPlanner(None)
        self.local_planner = DynamicWindowApproach({
            'max_speed': 0.5, 'min_speed': 0.0,
            'max_yawrate': 1.0, 'max_accel': 0.5,
            'max_dyawrate': 1.5, 'v_resolution': 0.1,
            'yawrate_resolution': 0.1, 'dt': 0.1,
            'predict_time': 3.0, 'to_goal_cost_gain': 1.0,
            'speed_cost_gain': 1.0, 'obstacle_cost_gain': 1.0
        })

        # Navigation state
        self.current_map = None
        self.current_scan = None
        self.goal_pose = None
        self.navigation_active = False

    def map_callback(self, msg):
        """Handle map updates"""
        self.current_map = self.ros_map_to_occupancy_grid(msg)
        self.global_planner.grid = self.current_map

    def scan_callback(self, msg):
        """Handle laser scan updates"""
        self.current_scan = msg

    def goal_callback(self, msg):
        """Handle new goal"""
        self.goal_pose = msg.pose
        self.navigation_active = True
        self.execute_navigation()

    def execute_navigation(self):
        """Main navigation execution loop"""
        while self.navigation_active and rclpy.ok():
            if self.current_map is None or self.goal_pose is None:
                continue

            # Get current robot pose
            current_pose = self.get_robot_pose()

            # Plan global path
            global_path = self.global_planner.plan_path(
                (current_pose.position.x, current_pose.position.y),
                (self.goal_pose.position.x, self.goal_pose.position.y)
            )

            if global_path is not None:
                self.path_pub.publish(self.path_to_ros_msg(global_path))

                # Execute local navigation along path
                self.follow_path(global_path)

            # Check if goal reached
            distance_to_goal = self.calculate_distance(
                current_pose, self.goal_pose
            )
            if distance_to_goal < 0.5:  # 0.5m tolerance
                self.navigation_active = False
                self.get_logger().info("Goal reached!")

    def follow_path(self, path):
        """Follow a planned path using local planner"""
        # Convert path to waypoints
        waypoints = self.discretize_path(path)

        for waypoint in waypoints:
            # Get current state
            current_state = self.get_robot_state()

            # Plan local trajectory to waypoint
            obstacles = self.extract_obstacles_from_scan()
            local_traj = self.local_planner.plan_local_path(
                current_state, waypoint, obstacles
            )

            if local_traj is not None:
                # Execute trajectory
                self.execute_trajectory(local_traj)

    def get_robot_pose(self):
        """Get current robot pose from TF"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time()
            )
            return transform.transform
        except:
            return None
```

## Navigation Safety and Recovery

### Recovery Behaviors

```python
class NavigationRecovery:
    def __init__(self, robot_controller):
        self.robot = robot_controller
        self.recovery_behaviors = {
            'clear_costmap': self.clear_costmap_recovery,
            'rotate_in_place': self.rotate_recovery,
            'move_backward': self.move_backward_recovery,
            'wait_and_retry': self.wait_recovery
        }

    def execute_recovery(self, failure_type):
        """
        Execute appropriate recovery behavior
        """
        if failure_type in self.recovery_behaviors:
            self.recovery_behaviors[failure_type]()
        else:
            self.emergency_stop()

    def clear_costmap_recovery(self):
        """
        Clear costmap to handle stale obstacle data
        """
        # Service call to clear costmaps
        # This would typically be a ROS service call
        pass

    def rotate_recovery(self):
        """
        Rotate in place to get better sensor data
        """
        current_yaw = self.robot.get_yaw()
        target_yaw = current_yaw + np.pi / 2  # Rotate 90 degrees

        # Rotate to new orientation
        self.robot.rotate_to_yaw(target_yaw, speed=0.5)

    def move_backward_recovery(self):
        """
        Move backward to get out of tight spaces
        """
        self.robot.move_backward(distance=0.3, speed=0.2)

    def wait_recovery(self):
        """
        Wait for dynamic obstacles to clear
        """
        import time
        time.sleep(2.0)  # Wait 2 seconds

    def emergency_stop(self):
        """
        Emergency stop behavior
        """
        self.robot.stop()
        self.robot.set_control_mode('safe')
```

## Multi-Robot Navigation

### Coordination and Communication

```python
class MultiRobotNavigator:
    def __init__(self, robot_id, total_robots):
        self.robot_id = robot_id
        self.total_robots = total_robots
        self.other_robots = {}  # Store info about other robots
        self.communication_range = 10.0  # meters

    def coordinate_navigation(self, own_path, other_robot_positions):
        """
        Coordinate navigation with other robots
        """
        # Update other robot information
        for robot_id, position in other_robot_positions.items():
            if robot_id != self.robot_id:
                self.other_robots[robot_id] = {
                    'position': position,
                    'timestamp': rospy.Time.now()
                }

        # Check for potential conflicts
        conflicts = self.detect_conflicts(own_path)

        if conflicts:
            # Negotiate path adjustments
            adjusted_path = self.negotiate_path(own_path, conflicts)
            return adjusted_path

        return own_path

    def detect_conflicts(self, own_path):
        """
        Detect potential conflicts with other robots
        """
        conflicts = []

        for robot_id, robot_info in self.other_robot_positions.items():
            if robot_id == self.robot_id:
                continue

            # Predict other robot's path
            predicted_path = self.predict_robot_path(robot_info)

            # Check for path intersections
            intersection = self.find_path_intersection(own_path, predicted_path)
            if intersection:
                conflicts.append({
                    'robot_id': robot_id,
                    'intersection': intersection,
                    'time': self.estimate_conflict_time(own_path, predicted_path, intersection)
                })

        return conflicts

    def negotiate_path(self, own_path, conflicts):
        """
        Negotiate path adjustments to avoid conflicts
        """
        # Simple priority-based negotiation
        # Robots with lower IDs have higher priority
        if self.robot_id == min(self.other_robots.keys()):
            return own_path  # Keep original path
        else:
            # Adjust path to avoid conflicts
            adjusted_path = self.adjust_path_for_conflicts(own_path, conflicts)
            return adjusted_path

    def adjust_path_for_conflicts(self, original_path, conflicts):
        """
        Adjust path to avoid detected conflicts
        """
        adjusted_path = original_path.copy()

        for conflict in conflicts:
            # Find waypoints near conflict
            conflict_idx = self.find_closest_waypoint(original_path, conflict['intersection'])

            # Modify path around conflict
            detour = self.calculate_detour(
                original_path[conflict_idx],
                conflict['robot_id']
            )

            # Insert detour into path
            adjusted_path = self.insert_detour(adjusted_path, detour, conflict_idx)

        return adjusted_path
```

## Advanced Navigation Techniques

### Learning-based Navigation

```python
import torch
import torch.nn as nn

class NavigationCNN(nn.Module):
    def __init__(self, num_actions=4):
        super().__init__()
        # CNN for processing sensor data (e.g., laser scan, camera)
        self.sensor_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(64)
        )

        # Combine with goal information
        self.fusion_layer = nn.Sequential(
            nn.Linear(64 + 2, 128),  # +2 for goal relative position
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Output navigation actions
        self.action_head = nn.Linear(64, num_actions)

    def forward(self, sensor_data, goal_relative_pos):
        # Process sensor data
        sensor_features = self.sensor_encoder(sensor_data.unsqueeze(0))
        sensor_features = sensor_features.view(1, -1)

        # Combine with goal information
        combined_input = torch.cat([sensor_features, goal_relative_pos.unsqueeze(0)], dim=1)

        # Process through fusion layer
        fused_features = self.fusion_layer(combined_input)

        # Output action probabilities
        action_probs = torch.softmax(self.action_head(fused_features), dim=1)

        return action_probs

class LearningBasedNavigator:
    def __init__(self):
        self.model = NavigationCNN()
        self.model.load_state_dict(torch.load('navigation_model.pth'))
        self.model.eval()

    def navigate(self, sensor_data, goal_pos, current_pos):
        """
        Navigate using learned policy
        """
        # Preprocess sensor data
        sensor_tensor = torch.tensor(sensor_data, dtype=torch.float32)

        # Calculate relative goal position
        goal_relative = torch.tensor([
            goal_pos[0] - current_pos[0],
            goal_pos[1] - current_pos[1]
        ], dtype=torch.float32)

        # Get action from learned policy
        with torch.no_grad():
            action_probs = self.model(sensor_tensor, goal_relative)

        # Select action (could be epsilon-greedy, etc.)
        action = torch.argmax(action_probs).item()

        return self.action_to_command(action)

    def action_to_command(self, action):
        """
        Convert discrete action to continuous command
        """
        # Define mapping from actions to commands
        # 0: turn left, 1: turn right, 2: go forward, 3: stop
        commands = {
            0: Twist(linear=Vector3(x=0.0, y=0.0, z=0.0),
                    angular=Vector3(x=0.0, y=0.0, z=0.5)),
            1: Twist(linear=Vector3(x=0.0, y=0.0, z=0.0),
                    angular=Vector3(x=0.0, y=0.0, z=-0.5)),
            2: Twist(linear=Vector3(x=0.5, y=0.0, z=0.0),
                    angular=Vector3(x=0.0, y=0.0, z=0.0)),
            3: Twist(linear=Vector3(x=0.0, y=0.0, z=0.0),
                    angular=Vector3(x=0.0, y=0.0, z=0.0))
        }

        return commands[action]
```

## Best Practices

### Navigation System Design Guidelines

1. **Hierarchical Planning**: Use global planning for route finding and local planning for obstacle avoidance
2. **Sensor Fusion**: Combine multiple sensors for robust environment perception
3. **Safety First**: Always implement collision avoidance and emergency stops
4. **Recovery Behaviors**: Plan for common failure scenarios
5. **Real-time Performance**: Ensure navigation algorithms meet timing requirements
6. **Map Management**: Keep maps updated and handle dynamic obstacles
7. **Localization Accuracy**: Maintain accurate pose estimation
8. **Path Optimization**: Smooth and optimize paths for efficient navigation

### Implementation Tips

- Use appropriate coordinate frames consistently
- Implement proper error handling and fallback behaviors
- Test navigation in simulation before real-world deployment
- Consider computational constraints when selecting algorithms
- Implement proper logging for debugging navigation issues
- Plan for different terrain types and environmental conditions
- Validate navigation performance under various obstacle densities
- Document navigation parameters and their effects on performance