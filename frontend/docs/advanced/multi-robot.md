---
sidebar_position: 6
---

# Multi-Robot Systems

Multi-robot systems involve the coordination and control of multiple robots to achieve common goals. This section covers the fundamental concepts, coordination strategies, and implementation techniques for effective multi-robot cooperation.

## Multi-Robot System Architectures

### Centralized vs. Decentralized Control

Multi-robot systems can be architected in different ways depending on the application requirements:

```python
class MultiRobotSystem:
    def __init__(self, num_robots):
        self.num_robots = num_robots
        self.robots = [Robot(i) for i in range(num_robots)]
        self.communication_graph = self.build_communication_graph()

    def centralized_control(self, global_task):
        """
        Centralized control architecture
        """
        # Central coordinator makes all decisions
        coordinator = CentralizedCoordinator(self.robots)
        robot_assignments = coordinator.assign_tasks(global_task)

        # Execute coordinated plan
        for robot_id, task in robot_assignments.items():
            self.robots[robot_id].execute_task(task)

    def decentralized_control(self, global_task):
        """
        Decentralized control architecture
        """
        # Each robot makes local decisions based on communication
        for robot in self.robots:
            local_task = robot.determine_local_task(global_task, self.get_neighbor_info(robot.id))
            robot.execute_task(local_task)

    def hybrid_control(self, global_task):
        """
        Hybrid control architecture
        """
        # Combine centralized and decentralized approaches
        # High-level task allocation is centralized
        # Low-level execution is decentralized
        task_allocation = self.centralized_task_allocation(global_task)

        for robot_id, task in task_allocation.items():
            # Local coordination with neighbors
            self.robots[robot_id].execute_coordinated_task(
                task, self.get_neighbor_info(robot_id)
            )
```

### Communication Topologies

Different communication patterns affect system performance and robustness:

```python
class CommunicationTopology:
    def __init__(self, num_robots):
        self.num_robots = num_robots
        self.connections = {}

    def build_star_topology(self, center_robot=0):
        """
        Star topology: all robots communicate through central robot
        """
        self.connections = {i: [center_robot] if i != center_robot else list(range(self.num_robots))
                           for i in range(self.num_robots)}
        return self.connections

    def build_ring_topology(self):
        """
        Ring topology: each robot communicates with adjacent robots
        """
        self.connections = {}
        for i in range(self.num_robots):
            neighbors = [(i-1) % self.num_robots, (i+1) % self.num_robots]
            self.connections[i] = neighbors
        return self.connections

    def build_mesh_topology(self, connection_radius=5.0):
        """
        Mesh topology: robots communicate based on physical proximity
        """
        # This would be based on actual robot positions
        self.connections = {i: [] for i in range(self.num_robots)}

        # Simulate positions and connect based on distance
        positions = self.get_robot_positions()
        for i in range(self.num_robots):
            for j in range(i+1, self.num_robots):
                if self.distance(positions[i], positions[j]) < connection_radius:
                    self.connections[i].append(j)
                    self.connections[j].append(i)

        return self.connections

    def get_communication_range(self, robot_id):
        """
        Get robots within communication range
        """
        return self.connections.get(robot_id, [])
```

## Task Allocation and Assignment

### Market-Based Task Allocation

Market-based approaches use economic principles for task allocation:

```python
class MarketBasedAllocator:
    def __init__(self, robots, tasks):
        self.robots = robots
        self.tasks = tasks
        self.bids = {}

    def run_auction(self):
        """
        Run task auction process
        """
        unassigned_tasks = self.tasks.copy()
        robot_assignments = {robot.id: [] for robot in self.robots}

        while unassigned_tasks:
            # Clear previous bids
            self.bids = {}

            # Each robot bids on tasks
            for robot in self.robots:
                if robot.id in robot_assignments and len(robot_assignments[robot.id]) < robot.max_tasks:
                    for task in unassigned_tasks:
                        bid = self.calculate_bid(robot, task)
                        if robot.id not in self.bids:
                            self.bids[robot.id] = {}
                        self.bids[robot.id][task.id] = bid

            # Assign tasks to highest bidders
            for task in unassigned_tasks.copy():
                best_robot = None
                best_bid = -float('inf')

                for robot in self.robots:
                    if robot.id in self.bids and task.id in self.bids[robot.id]:
                        if self.bids[robot.id][task.id] > best_bid:
                            best_bid = self.bids[robot.id][task.id]
                            best_robot = robot

                if best_robot:
                    robot_assignments[best_robot.id].append(task)
                    unassigned_tasks.remove(task)

        return robot_assignments

    def calculate_bid(self, robot, task):
        """
        Calculate bid value for a task
        """
        # Factors affecting bid: task difficulty, robot capability, distance, etc.
        distance_cost = self.calculate_distance(robot.position, task.location)
        capability_match = self.calculate_capability_match(robot, task)
        energy_cost = robot.estimate_energy_cost(task)

        # Bid = capability_match - (distance_cost + energy_cost)
        bid = capability_match - (distance_cost * 0.1 + energy_cost * 0.05)
        return bid
```

### Consensus-Based Allocation

Consensus algorithms ensure agreement among robots:

```python
class ConsensusBasedAllocator:
    def __init__(self, robots, communication_graph):
        self.robots = robots
        self.graph = communication_graph
        self.task_values = {robot.id: {} for robot in robots}

    def run_consensus_allocation(self, tasks):
        """
        Run consensus-based task allocation
        """
        # Initialize task values for each robot
        for robot in self.robots:
            for task in tasks:
                self.task_values[robot.id][task.id] = self.evaluate_task(robot, task)

        # Run consensus iterations
        max_iterations = 100
        for iteration in range(max_iterations):
            new_values = {robot.id: {} for robot in self.robots}

            for robot in self.robots:
                neighbors = self.graph.get_communication_range(robot.id)

                for task in tasks:
                    # Average values from neighbors
                    neighbor_values = [self.task_values[n][task.id] for n in neighbors if n in self.task_values]
                    neighbor_values.append(self.task_values[robot.id][task.id])

                    new_values[robot.id][task.id] = sum(neighbor_values) / len(neighbor_values)

            self.task_values = new_values

            # Check for convergence
            if self.check_convergence():
                break

        # Assign tasks based on final values
        return self.assign_tasks_from_values()

    def evaluate_task(self, robot, task):
        """
        Evaluate how suitable a task is for a robot
        """
        # Consider robot capabilities, task requirements, distance, etc.
        capability_score = self.calculate_capability_score(robot, task)
        distance_penalty = self.calculate_distance_penalty(robot, task)
        energy_factor = robot.energy_level / 100.0  # Normalize energy level

        return capability_score - distance_penalty + energy_factor

    def check_convergence(self):
        """
        Check if consensus values have converged
        """
        # Implementation to check if values have stabilized
        return False  # Simplified
```

## Formation Control

### Leader-Follower Formation

Leader-follower formations maintain specific geometric patterns:

```python
class LeaderFollowerFormation:
    def __init__(self, leader, followers, formation_pattern):
        self.leader = leader
        self.followers = followers
        self.pattern = formation_pattern  # e.g., line, wedge, diamond
        self.formation_positions = self.calculate_formation_positions()

    def calculate_formation_positions(self):
        """
        Calculate desired positions relative to leader
        """
        positions = {}

        for i, follower in enumerate(self.followers):
            # Calculate offset based on formation pattern
            offset = self.pattern.get_offset(i, len(self.followers))
            positions[follower.id] = offset

        return positions

    def maintain_formation(self):
        """
        Maintain formation while leader moves
        """
        leader_pose = self.leader.get_pose()
        leader_vel = self.leader.get_velocity()

        for follower in self.followers:
            # Calculate desired position
            desired_offset = self.formation_positions[follower.id]
            desired_pos = self.transform_pose(leader_pose, desired_offset)

            # Calculate desired velocity
            desired_vel = self.transform_velocity(leader_vel, desired_offset)

            # Control follower to desired state
            follower.follow_trajectory(desired_pos, desired_vel)

    def transform_pose(self, leader_pose, offset):
        """
        Transform offset to world coordinates based on leader pose
        """
        # Apply rotation and translation
        cos_yaw = np.cos(leader_pose.orientation.z)
        sin_yaw = np.sin(leader_pose.orientation.z)

        # Rotate offset by leader's orientation
        rotated_x = offset.x * cos_yaw - offset.y * sin_yaw
        rotated_y = offset.x * sin_yaw + offset.y * cos_yaw

        # Add to leader's position
        world_x = leader_pose.position.x + rotated_x
        world_y = leader_pose.position.y + rotated_y

        return np.array([world_x, world_y])

    def reconfigure_formation(self, new_pattern):
        """
        Reconfigure formation pattern dynamically
        """
        self.pattern = new_pattern
        self.formation_positions = self.calculate_formation_positions()
```

### Virtual Structure Formation

Virtual structure formations maintain rigid body-like behavior:

```python
class VirtualStructureFormation:
    def __init__(self, robots, structure_shape):
        self.robots = robots
        self.structure = structure_shape  # Define geometric shape
        self.virtual_pose = np.array([0, 0, 0])  # x, y, theta of virtual structure
        self.robot_offsets = self.calculate_robot_offsets()

    def calculate_robot_offsets(self):
        """
        Calculate fixed offsets of robots from virtual structure
        """
        offsets = {}
        for i, robot in enumerate(self.robots):
            # Get position in virtual structure
            pos_in_structure = self.structure.get_robot_position(i)
            offsets[robot.id] = pos_in_structure
        return offsets

    def update_virtual_motion(self, desired_motion):
        """
        Update virtual structure motion
        """
        self.virtual_pose += desired_motion * self.dt

    def control_robots(self):
        """
        Control each robot to maintain virtual structure
        """
        for robot in self.robots:
            # Calculate desired robot pose based on virtual structure
            robot_offset = self.robot_offsets[robot.id]
            desired_pose = self.transform_to_robot_pose(
                self.virtual_pose, robot_offset
            )

            # Control robot to desired pose
            robot.goto_pose(desired_pose)

    def transform_to_robot_pose(self, virtual_pose, offset):
        """
        Transform virtual structure pose to individual robot pose
        """
        vx, vy, vtheta = virtual_pose
        ox, oy = offset

        # Rotate offset by virtual structure orientation
        cos_t = np.cos(vtheta)
        sin_t = np.sin(vtheta)

        rotated_x = ox * cos_t - oy * sin_t
        rotated_y = ox * sin_t + oy * cos_t

        robot_x = vx + rotated_x
        robot_y = vy + rotated_y
        robot_theta = vtheta

        return np.array([robot_x, robot_y, robot_theta])
```

## Multi-Robot Path Planning

### Conflict-Based Search (CBS)

CBS is a complete algorithm for multi-robot path planning:

```python
class ConflictBasedSearch:
    def __init__(self, grid_map, robots_start, robots_goal):
        self.map = grid_map
        self.starts = robots_start
        self.goals = robots_goal
        self.num_robots = len(robots_start)

    def solve(self):
        """
        Solve multi-robot path planning using CBS
        """
        # High-level search for joint solution
        root = self.create_root_node()
        open_list = [root]

        while open_list:
            node = heapq.heappop(open_list)

            # Check for conflicts
            conflict = self.find_conflict(node.paths)

            if conflict is None:
                # No conflicts, solution found
                return node.paths

            # Add constraints for conflicting robots
            constraint1, constraint2 = self.create_constraints(conflict)

            # Create new nodes with constraints
            node1 = self.create_constrained_node(node, constraint1)
            node2 = self.create_constrained_node(node, constraint2)

            heapq.heappush(open_list, node1)
            heapq.heappush(open_list, node2)

        return None  # No solution found

    def create_root_node(self):
        """
        Create root node with individual paths
        """
        paths = []
        for i in range(self.num_robots):
            planner = AStarPlanner(self.map)
            path = planner.plan_path(self.starts[i], self.goals[i])
            paths.append(path)

        return CBSNode(paths=paths, constraints=[])

    def find_conflict(self, paths):
        """
        Find first conflict between paths
        """
        for t in range(max(len(path) for path in paths)):
            for i in range(self.num_robots):
                for j in range(i + 1, self.num_robots):
                    if t < len(paths[i]) and t < len(paths[j]):
                        pos_i = paths[i][t] if t < len(paths[i]) else paths[i][-1]
                        pos_j = paths[j][t] if t < len(paths[j]) else paths[j][-1]

                        if np.array_equal(pos_i, pos_j):
                            return {'timestep': t, 'robots': [i, j], 'location': pos_i}

                        # Edge conflicts (swapping positions)
                        if (t > 0 and t < len(paths[i]) and t < len(paths[j]) and
                            np.array_equal(paths[i][t-1], paths[j][t]) and
                            np.array_equal(paths[i][t], paths[j][t-1])):
                            return {
                                'timestep': t,
                                'robots': [i, j],
                                'edge': (paths[i][t-1], paths[i][t])
                            }

        return None

class CBSNode:
    def __init__(self, paths, constraints):
        self.paths = paths
        self.constraints = constraints
        self.cost = sum(len(path) for path in paths)

    def __lt__(self, other):
        return self.cost < other.cost
```

### Prioritized Planning

Prioritized planning assigns robots different priorities:

```python
class PrioritizedPlanner:
    def __init__(self, robots, starts, goals, priorities):
        self.robots = robots
        self.starts = starts
        self.goals = goals
        self.priorities = priorities  # Higher number = higher priority

    def plan_with_priorities(self):
        """
        Plan paths with robot priorities
        """
        # Sort robots by priority (highest first)
        sorted_indices = sorted(range(len(self.priorities)),
                              key=lambda i: self.priorities[i], reverse=True)

        paths = [None] * len(self.robots)
        occupied_positions = {}  # Track positions over time

        for robot_idx in sorted_indices:
            robot_start = self.starts[robot_idx]
            robot_goal = self.goals[robot_idx]

            # Plan path considering higher-priority robot paths
            path = self.plan_path_with_constraints(
                robot_start, robot_goal, occupied_positions
            )

            if path is None:
                return None  # No path found

            paths[robot_idx] = path

            # Update occupied positions
            for t, pos in enumerate(path):
                if t not in occupied_positions:
                    occupied_positions[t] = set()
                occupied_positions[t].add(tuple(pos))

        return paths

    def plan_path_with_constraints(self, start, goal, occupied_positions):
        """
        Plan path avoiding occupied positions
        """
        # Use A* with modified collision checking
        planner = AStarPlannerWithConstraints(occupied_positions)
        return planner.plan_path(start, goal)
```

## Coordination and Communication

### Message Passing for Coordination

```python
class RobotCommunication:
    def __init__(self, robot_id, communication_range=10.0):
        self.robot_id = robot_id
        self.range = communication_range
        self.message_queue = []
        self.neighbors = []

    def broadcast_message(self, message_type, data):
        """
        Broadcast message to all neighbors
        """
        message = {
            'sender': self.robot_id,
            'type': message_type,
            'data': data,
            'timestamp': time.time()
        }

        for neighbor in self.neighbors:
            self.send_message(neighbor, message)

    def send_message(self, recipient, message):
        """
        Send message to specific robot
        """
        # Simulate message passing
        recipient.receive_message(message)

    def receive_message(self, message):
        """
        Receive and process incoming message
        """
        self.message_queue.append(message)

        # Process based on message type
        if message['type'] == 'task_assignment':
            self.handle_task_assignment(message['data'])
        elif message['type'] == 'position_update':
            self.update_neighbor_position(message['sender'], message['data'])
        elif message['type'] == 'status_report':
            self.handle_status_report(message['sender'], message['data'])

    def synchronize_with_neighbors(self):
        """
        Synchronize information with neighbors
        """
        # Exchange state information
        my_state = self.get_current_state()
        self.broadcast_message('position_update', my_state)

        # Process received messages
        while self.message_queue:
            msg = self.message_queue.pop(0)
            self.process_message(msg)

    def process_message(self, message):
        """
        Process incoming message based on type
        """
        msg_type = message['type']
        sender = message['sender']
        data = message['data']

        if msg_type == 'position_update':
            self.update_neighbor_position(sender, data)
        elif msg_type == 'intent_broadcast':
            self.update_neighbor_intent(sender, data)
        elif msg_type == 'resource_request':
            self.handle_resource_request(sender, data)
```

### Distributed Coordination Protocols

```python
class DistributedCoordinator:
    def __init__(self, robot_id, total_robots):
        self.robot_id = robot_id
        self.total_robots = total_robots
        self.state = 'idle'
        self.leader = None
        self.group_id = robot_id  # Initially each robot is its own group

    def elect_leader(self):
        """
        Elect leader using distributed algorithm
        """
        # Simple ID-based election
        if self.robot_id == max(range(self.total_robots)):
            self.state = 'leader'
            self.leader = self.robot_id
            # Notify others
            self.broadcast_message('leader_elected', {'leader': self.robot_id})
        else:
            # Wait for leader election message
            self.state = 'follower'
            # Implement timeout and retry logic

    def form_coalition(self, task_requirements):
        """
        Form coalition for complex tasks
        """
        # Broadcast task requirements
        self.broadcast_message('task_request', {
            'task_id': self.generate_task_id(),
            'requirements': task_requirements,
            'originator': self.robot_id
        })

        # Wait for responses and form coalition
        potential_members = self.wait_for_responders(task_requirements)
        coalition = self.select_optimal_coalition(potential_members, task_requirements)

        return coalition

    def consensus_decision(self, proposal):
        """
        Reach consensus on proposal using distributed voting
        """
        # Broadcast proposal
        self.broadcast_message('proposal', {
            'proposal': proposal,
            'proposer': self.robot_id
        })

        # Collect votes
        votes = {}
        timeout = time.time() + 10.0  # 10 second timeout

        while time.time() < timeout and len(votes) < self.total_robots:
            # Process incoming votes
            vote = self.receive_vote()
            if vote:
                votes[vote['voter']] = vote['decision']

        # Count votes
        yes_votes = sum(1 for decision in votes.values() if decision == 'yes')
        total_votes = len(votes)

        return yes_votes > total_votes / 2  # Simple majority
```

## Multi-Robot Learning and Adaptation

### Multi-Agent Reinforcement Learning

```python
import torch
import torch.nn as nn
import numpy as np

class MultiAgentDQN(nn.Module):
    def __init__(self, num_agents, state_dim, action_dim):
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared network components
        self.shared_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

        # Individual agent networks
        self.agent_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim)
            ) for _ in range(num_agents)
        ])

    def forward(self, states):
        """
        Forward pass for all agents
        """
        shared_features = self.shared_encoder(states)

        q_values = []
        for i in range(self.num_agents):
            agent_q = self.agent_networks[i](shared_features)
            q_values.append(agent_q)

        return q_values

class MultiRobotMARL:
    def __init__(self, num_robots):
        self.num_robots = num_robots
        self.models = [MultiAgentDQN(num_robots, 50, 4) for _ in range(num_robots)]
        self.target_models = [MultiAgentDQN(num_robots, 50, 4) for _ in range(num_robots)]

    def get_joint_action(self, joint_state):
        """
        Get joint action for all robots
        """
        joint_q_values = []
        for i in range(self.num_robots):
            with torch.no_grad():
                q_vals = self.models[i](torch.tensor(joint_state, dtype=torch.float32))
                joint_q_values.append(q_vals[i])  # Agent i's Q-values

        # Convert to actions
        actions = []
        for q_vals in joint_q_values:
            action = torch.argmax(q_vals).item()
            actions.append(action)

        return actions
```

### Cooperative Learning

```python
class CooperativeLearningSystem:
    def __init__(self, robots):
        self.robots = robots
        self.shared_experience_buffer = []
        self.communication_graph = self.build_communication_graph()

    def share_experience(self, experiences):
        """
        Share experiences between robots
        """
        for robot in self.robots:
            neighbors = self.communication_graph[robot.id]

            # Share recent experiences with neighbors
            for neighbor_id in neighbors:
                neighbor = self.get_robot_by_id(neighbor_id)
                neighbor.receive_experience(experiences)

    def collaborative_training(self):
        """
        Train robots collaboratively
        """
        # Aggregate experiences from all robots
        all_experiences = []
        for robot in self.robots:
            all_experiences.extend(robot.local_experience_buffer)

        # Train each robot with combined data
        for robot in self.robots:
            robot.train_on_shared_data(all_experiences)

    def adapt_to_environment(self, environment_changes):
        """
        Adapt to environmental changes collaboratively
        """
        # Detect changes collectively
        change_detections = []
        for robot in self.robots:
            detection = robot.detect_environment_change(environment_changes)
            change_detections.append(detection)

        # Share and aggregate change information
        aggregated_changes = self.aggregate_changes(change_detections)

        # Adapt collectively
        for robot in self.robots:
            robot.adapt_behavior(aggregated_changes)
```

## Safety and Robustness

### Fault Tolerance

```python
class FaultTolerantMultiRobotSystem:
    def __init__(self, robots):
        self.robots = robots
        self.status_monitor = StatusMonitor(robots)
        self.backup_plans = {}

    def monitor_robot_status(self):
        """
        Monitor robot health and status
        """
        for robot in self.robots:
            status = self.status_monitor.check_robot(robot)

            if status['health'] < 0.3:  # Critical health
                self.handle_robot_failure(robot.id)

    def handle_robot_failure(self, failed_robot_id):
        """
        Handle robot failure and reassign tasks
        """
        # Identify tasks assigned to failed robot
        failed_tasks = self.get_tasks_for_robot(failed_robot_id)

        # Reassign tasks to healthy robots
        for task in failed_tasks:
            new_assignee = self.find_alternative_robot(task, exclude=[failed_robot_id])
            if new_assignee:
                self.assign_task(new_assignee, task)

        # Update formation or coordination as needed
        self.update_formation_after_failure(failed_robot_id)

    def redundancy_management(self):
        """
        Manage redundancy for critical tasks
        """
        critical_tasks = self.get_critical_tasks()

        for task in critical_tasks:
            assign_backup_robots = self.select_backup_robots(task)
            task.set_backup_robots(assign_backup_robots)

    def graceful_degradation(self):
        """
        Maintain functionality with reduced capabilities
        """
        active_robots = self.get_active_robots()
        required_robots = self.calculate_minimum_robots_for_tasks()

        if len(active_robots) < required_robots:
            # Reduce task complexity or scope
            self.reduce_task_complexity()
            self.reallocate_remaining_tasks()
```

### Safety Protocols

```python
class MultiRobotSafetySystem:
    def __init__(self, robot_radius=0.3):
        self.robot_radius = robot_radius
        self.safety_margin = 0.5
        self.min_separation = robot_radius * 2 + self.safety_margin

    def check_collision_avoidance(self, robot_positions, robot_velocities):
        """
        Check for potential collisions between robots
        """
        for i in range(len(robot_positions)):
            for j in range(i + 1, len(robot_positions)):
                pos_i = robot_positions[i]
                pos_j = robot_positions[j]

                distance = np.linalg.norm(pos_i - pos_j)

                if distance < self.min_separation:
                    # Calculate collision avoidance maneuver
                    avoidance_vector = self.calculate_avoidance_vector(pos_i, pos_j)
                    robot_velocities[i] += avoidance_vector * 0.1
                    robot_velocities[j] -= avoidance_vector * 0.1

        return robot_velocities

    def calculate_avoidance_vector(self, pos1, pos2):
        """
        Calculate avoidance vector to prevent collision
        """
        direction = pos1 - pos2
        distance = np.linalg.norm(direction)

        if distance < 0.01:  # Avoid division by zero
            # Random perpendicular direction
            return np.array([-direction[1], direction[0]])

        unit_direction = direction / distance
        return unit_direction

    def enforce_safety_zones(self, robot_id, position):
        """
        Enforce safety zones around robots
        """
        # Create safety zone around position
        safety_zone = {
            'center': position,
            'radius': self.min_separation,
            'owner': robot_id
        }

        # Check conflicts with other safety zones
        conflicts = self.check_zone_conflicts(safety_zone)

        if conflicts:
            return self.resolve_zone_conflicts(safety_zone, conflicts)

        return True
```

## Implementation with ROS 2

### Multi-Robot Communication Layer

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32
from geometry_msgs.msg import PoseStamped, Twist
from multi_robot_msgs.msg import RobotStatus, TaskAssignment

class MultiRobotCoordinator(Node):
    def __init__(self):
        super().__init__('multi_robot_coordinator')

        # Robot identification
        self.robot_id = self.declare_parameter('robot_id', 0).value
        self.total_robots = self.declare_parameter('total_robots', 3).value

        # Publishers for inter-robot communication
        self.status_pub = {}
        self.task_pub = {}

        for i in range(self.total_robots):
            if i != self.robot_id:
                self.status_pub[i] = self.create_publisher(
                    RobotStatus, f'/robot_{i}/status', 10
                )
                self.task_pub[i] = self.create_publisher(
                    TaskAssignment, f'/robot_{i}/task_assignment', 10
                )

        # Subscribers for receiving information from other robots
        self.status_subs = {}
        for i in range(self.total_robots):
            if i != self.robot_id:
                self.status_subs[i] = self.create_subscription(
                    RobotStatus, f'/robot_{self.robot_id}/from_robot_{i}/status',
                    self.create_status_callback(i), 10
                )

        # Task management
        self.task_allocator = MarketBasedAllocator([], [])
        self.current_tasks = {}

    def create_status_callback(self, robot_id):
        """
        Create callback function for specific robot
        """
        def callback(msg):
            self.process_robot_status(robot_id, msg)
        return callback

    def broadcast_status(self, status_info):
        """
        Broadcast status to all other robots
        """
        msg = RobotStatus()
        msg.robot_id = self.robot_id
        msg.status_info = status_info
        msg.timestamp = self.get_clock().now().to_msg()

        for other_robot_id in self.status_pub:
            self.status_pub[other_robot_id].publish(msg)

    def coordinate_task_allocation(self, global_tasks):
        """
        Coordinate task allocation among robots
        """
        # Use auction-based allocation
        assignments = self.task_allocator.run_auction(global_tasks)

        # Publish assignments to respective robots
        for robot_id, task_list in assignments.items():
            if robot_id != self.robot_id:
                task_msg = TaskAssignment()
                task_msg.tasks = task_list
                self.task_pub[robot_id].publish(task_msg)

    def process_robot_status(self, robot_id, status_msg):
        """
        Process status information from another robot
        """
        # Update internal state with received information
        self.other_robot_states[robot_id] = {
            'position': status_msg.position,
            'status': status_msg.status,
            'tasks': status_msg.assigned_tasks,
            'timestamp': status_msg.timestamp
        }

        # Check for coordination opportunities
        self.check_coordination_opportunities(robot_id, status_msg)
```

## Best Practices

### Multi-Robot System Design Guidelines

1. **Scalability**: Design systems that can handle varying numbers of robots
2. **Communication Efficiency**: Minimize communication overhead
3. **Fault Tolerance**: Plan for robot failures and maintain system functionality
4. **Safety**: Implement collision avoidance and safe operation protocols
5. **Load Balancing**: Distribute tasks evenly among robots
6. **Consistency**: Maintain consistent state information across robots
7. **Real-time Performance**: Ensure coordination algorithms meet timing requirements
8. **Adaptability**: Design systems that can adapt to changing conditions

### Implementation Tips

- Use standardized message formats for inter-robot communication
- Implement proper error handling and recovery mechanisms
- Test with varying numbers of robots to ensure scalability
- Consider communication delays and packet losses in real deployments
- Implement proper logging and monitoring for debugging
- Plan for graceful degradation when robots fail
- Use simulation extensively before real-world deployment
- Document coordination protocols and their assumptions
- Consider computational constraints on individual robots