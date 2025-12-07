---
sidebar_position: 4
---

# Robotic Manipulation

Robotic manipulation involves the control of robot arms and end-effectors to interact with objects in the environment. This section covers the fundamental concepts, techniques, and algorithms for dexterous manipulation.

## Manipulator Kinematics

### Forward Kinematics

Forward kinematics calculates the end-effector position and orientation from joint angles:

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

class ManipulatorKinematics:
    def __init__(self, dh_parameters):
        """
        Initialize with Denavit-Hartenberg parameters
        dh_parameters: list of [alpha, a, d, theta_offset] for each joint
        """
        self.dh_params = dh_parameters
        self.num_joints = len(dh_parameters)

    def dh_transform(self, alpha, a, d, theta):
        """
        Denavit-Hartenberg transformation matrix
        """
        return np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])

    def forward_kinematics(self, joint_angles):
        """
        Calculate end-effector pose from joint angles
        """
        if len(joint_angles) != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joint angles, got {len(joint_angles)}")

        T = np.eye(4)  # Start with identity matrix

        for i in range(self.num_joints):
            alpha, a, d, theta_offset = self.dh_params[i]
            theta = joint_angles[i] + theta_offset
            T_link = self.dh_transform(alpha, a, d, theta)
            T = T @ T_link

        # Extract position and orientation
        position = T[:3, 3]
        orientation = T[:3, :3]

        return {
            'position': position,
            'orientation': orientation,
            'transform': T
        }

    def jacobian(self, joint_angles):
        """
        Calculate geometric Jacobian matrix
        """
        J = np.zeros((6, self.num_joints))  # 6 DOF (3 pos + 3 rot)

        # Get all link transforms
        transforms = []
        T_cumulative = np.eye(4)

        for i in range(self.num_joints):
            alpha, a, d, theta_offset = self.dh_params[i]
            theta = joint_angles[i] + theta_offset
            T_link = self.dh_transform(alpha, a, d, theta)
            T_cumulative = T_cumulative @ T_link
            transforms.append(T_cumulative.copy())

        # End-effector position
        end_pos = transforms[-1][:3, 3]

        for i in range(self.num_joints):
            # Z-axis of joint i (in world coordinates)
            z_i = transforms[i][:3, 2]
            # Position of joint i (in world coordinates)
            pos_i = transforms[i][:3, 3]

            # Linear velocity component
            J[:3, i] = np.cross(z_i, (end_pos - pos_i))
            # Angular velocity component
            J[3:, i] = z_i

        return J
```

### Inverse Kinematics

Inverse kinematics solves for joint angles that achieve a desired end-effector pose:

```python
class InverseKinematics:
    def __init__(self, manipulator_kinematics):
        self.kinematics = manipulator_kinematics

    def jacobian_inverse_kinematics(self, target_pose, initial_joints, max_iter=100, tolerance=1e-4):
        """
        Solve IK using Jacobian transpose/pseudo-inverse method
        """
        joints = initial_joints.copy()
        target_pos = target_pose['position']
        target_rot = target_pose['orientation']

        for i in range(max_iter):
            # Forward kinematics
            current_pose = self.kinematics.forward_kinematics(joints)
            current_pos = current_pose['position']
            current_rot = current_pose['orientation']

            # Calculate position error
            pos_error = target_pos - current_pos

            # Calculate orientation error (using rotation matrix difference)
            rot_error_matrix = target_rot @ current_rot.T - np.eye(3)
            rot_error = self.rotation_matrix_to_vector(rot_error_matrix)

            # Combine position and orientation errors
            error = np.concatenate([pos_error, rot_error])

            # Check convergence
            if np.linalg.norm(error) < tolerance:
                break

            # Calculate Jacobian
            J = self.kinematics.jacobian(joints)

            # Solve for joint velocity
            # Use damped least squares to handle singularities
            damping = 0.01
            JJT = J @ J.T
            I = np.eye(JJT.shape[0])
            J_pinv = J.T @ np.linalg.inv(JJT + damping**2 * I)

            joint_vel = J_pinv @ error
            joints = joints + 0.1 * joint_vel  # Integration step

        return joints

    def rotation_matrix_to_vector(self, R):
        """
        Convert rotation matrix to rotation vector
        """
        # Use Rodrigues formula
        trace = np.trace(R)
        if trace > 3 - 1e-6:  # Identity matrix
            return np.zeros(3)

        angle = np.arccos((trace - 1) / 2)
        if abs(angle) < 1e-6:  # Small angle approximation
            return 0.5 * np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]])

        scale = angle / (2 * np.sin(angle))
        rot_vec = scale * np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]])
        return rot_vec

    def analytical_ik_6dof(self, target_pose):
        """
        Analytical solution for 6-DOF manipulator (e.g., PUMA-style)
        """
        # This is a simplified example - actual implementation depends on specific robot geometry
        px, py, pz = target_pose['position']
        R = target_pose['orientation']

        # Calculate wrist center position
        # Assuming spherical wrist (3 last joints intersect at wrist center)
        a6 = 0  # Tool offset
        wrist_center = np.array([px, py, pz]) - a6 * R[:3, 2]

        # Position inverse kinematics (first 3 joints)
        # This involves solving for shoulder, elbow, and wrist angles
        # Implementation depends on specific robot geometry

        # Orientation inverse kinematics (last 3 joints)
        # Solve for wrist angles to achieve desired orientation

        # Return joint angles (simplified)
        return np.zeros(6)  # Placeholder
```

## Grasp Planning and Execution

### Grasp Pose Generation

```python
class GraspPlanner:
    def __init__(self, robot_model, gripper_model):
        self.robot = robot_model
        self.gripper = gripper_model
        self.collision_checker = CollisionChecker()

    def generate_grasps(self, object_mesh, approach_direction='top'):
        """
        Generate potential grasp poses for an object
        """
        grasps = []

        # Sample potential grasp points on object surface
        surface_points = self.sample_surface_points(object_mesh)

        for point in surface_points:
            # Calculate approach direction based on surface normal
            normal = self.calculate_surface_normal(object_mesh, point)

            # Generate grasp orientations
            grasp_poses = self.generate_grasp_orientations(
                point, normal, approach_direction
            )

            for pose in grasp_poses:
                # Check grasp quality
                quality = self.evaluate_grasp_quality(pose, object_mesh)

                if quality > 0.5:  # Threshold for acceptable grasp
                    grasp = {
                        'pose': pose,
                        'quality': quality,
                        'approach_direction': self.calculate_approach_direction(pose)
                    }
                    grasps.append(grasp)

        # Sort by quality
        grasps.sort(key=lambda x: x['quality'], reverse=True)

        return grasps

    def evaluate_grasp_quality(self, grasp_pose, object_mesh):
        """
        Evaluate grasp stability and quality
        """
        # Check force closure (ability to resist external forces)
        contact_points = self.calculate_contact_points(grasp_pose)

        # Check if grasp is force-closure
        force_closure = self.check_force_closure(contact_points, object_mesh)

        # Check grasp wrench space
        wrench_space = self.calculate_grasp_wrench_space(contact_points)

        # Check collision with environment
        collision_free = self.check_grasp_collision(grasp_pose)

        # Combine metrics into quality score
        quality = (
            0.4 * force_closure +
            0.3 * wrench_space +
            0.3 * (1.0 if collision_free else 0.0)
        )

        return quality

    def plan_grasp_trajectory(self, object_pose, grasp_pose, pregrasp_distance=0.1):
        """
        Plan trajectory to execute grasp
        """
        # Calculate pre-grasp pose (approach position)
        approach_direction = grasp_pose['orientation'][:, 2]  # Z-axis is approach direction
        pregrasp_pose = grasp_pose.copy()
        pregrasp_pose['position'] = (
            grasp_pose['position'] - pregrasp_distance * approach_direction
        )

        # Plan path to pre-grasp
        approach_path = self.plan_path_to_pose(pregrasp_pose)

        # Plan to grasp pose
        grasp_path = self.linear_interpolation(pregrasp_pose, grasp_pose)

        # Combine trajectories
        full_trajectory = {
            'approach': approach_path,
            'grasp': grasp_path,
            'lift': self.generate_lift_trajectory(grasp_pose)
        }

        return full_trajectory
```

## Force Control and Compliance

### Impedance Control

Impedance control allows robots to behave like springs when interacting with the environment:

```python
class ImpedanceController:
    def __init__(self, mass, damping, stiffness):
        self.M = mass      # Desired mass matrix
        self.D = damping   # Desired damping matrix
        self.K = stiffness # Desired stiffness matrix

    def compute_impedance_force(self, position_error, velocity_error, desired_force=None):
        """
        Compute impedance control force
        """
        if desired_force is None:
            desired_force = np.zeros_like(position_error)

        # Impedance control law: F = M*(xdd_des - xdd) + D*(xd_des - xd) + K*(x_des - x)
        impedance_force = (
            self.K @ position_error +
            self.D @ velocity_error +
            desired_force
        )

        return impedance_force

    def cartesian_impedance_control(self, current_pose, desired_pose,
                                   current_twist, desired_twist):
        """
        Cartesian impedance control for end-effector
        """
        # Calculate pose error (in Cartesian space)
        pos_error = desired_pose['position'] - current_pose['position']

        # Calculate orientation error
        R_current = current_pose['orientation']
        R_desired = desired_pose['orientation']
        R_error = R_desired @ R_current.T
        rot_error = self.rotation_matrix_to_vector(R_error)

        pose_error = np.concatenate([pos_error, rot_error])
        twist_error = desired_twist - current_twist

        # Compute impedance force
        F_impedance = self.compute_impedance_force(pose_error, twist_error)

        return F_impedance

    def admittance_control(self, applied_force, dt):
        """
        Admittance control - integrates force to get motion
        """
        # Admittance: x_ddot = A * F
        # Where A is admittance matrix (inverse of impedance)
        A = np.linalg.inv(self.M)  # Simplified - typically A = inv(M)

        acceleration = A @ applied_force
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

        return self.position, self.velocity
```

## Manipulation Strategies

### Pick and Place Operations

```python
class PickPlacePlanner:
    def __init__(self, robot_controller, grasp_planner):
        self.robot = robot_controller
        self.grasp_planner = grasp_planner
        self.trajectory_generator = TrajectoryGenerator()

    def execute_pick_place(self, object_pose, target_pose, approach_height=0.2):
        """
        Execute complete pick and place operation
        """
        # 1. Plan approach trajectory
        approach_traj = self.plan_approach_trajectory(object_pose, approach_height)

        # 2. Execute approach
        self.robot.execute_trajectory(approach_traj)

        # 3. Plan grasp trajectory
        grasp_poses = self.grasp_planner.generate_grasps(object_pose)
        if not grasp_poses:
            raise Exception("No valid grasps found")

        best_grasp = grasp_poses[0]
        grasp_traj = self.grasp_planner.plan_grasp_trajectory(
            object_pose, best_grasp['pose']
        )

        # 4. Execute grasp
        self.execute_grasp(grasp_traj)

        # 5. Plan lift trajectory
        lift_traj = self.plan_lift_trajectory(best_grasp['pose'], approach_height)
        self.robot.execute_trajectory(lift_traj)

        # 6. Plan transport to target
        transport_traj = self.plan_transport_trajectory(target_pose, approach_height)
        self.robot.execute_trajectory(transport_traj)

        # 7. Plan placement
        place_traj = self.plan_placement_trajectory(target_pose)
        self.robot.execute_trajectory(place_traj)

        # 8. Release object
        self.robot.open_gripper()

        # 9. Retract
        retract_traj = self.plan_retract_trajectory(target_pose, approach_height)
        self.robot.execute_trajectory(retract_traj)

    def plan_approach_trajectory(self, object_pose, approach_height):
        """
        Plan approach trajectory above object
        """
        # Calculate approach point above object
        approach_pos = object_pose['position'].copy()
        approach_pos[2] += approach_height

        approach_pose = {
            'position': approach_pos,
            'orientation': object_pose['orientation']  # Maintain same orientation
        }

        # Plan path from current position to approach position
        current_pose = self.robot.get_current_pose()
        trajectory = self.trajectory_generator.linear_interpolation(
            current_pose, approach_pose
        )

        return trajectory

    def execute_grasp(self, grasp_trajectory):
        """
        Execute grasp with force control
        """
        # Follow approach trajectory
        self.robot.execute_trajectory(grasp_trajectory['approach'])

        # Close gripper with force control
        self.robot.close_gripper_with_force_control(
            target_force=20.0,  # 20N grasp force
            max_effort=50.0
        )

        # Verify grasp success
        if not self.verify_grasp():
            raise Exception("Grasp failed")
```

## Multi-fingered Hand Control

### Grasp Synthesis

```python
class MultiFingeredHandController:
    def __init__(self, num_fingers=5):
        self.num_fingers = num_fingers
        self.finger_controllers = [FingerController(i) for i in range(num_fingers)]

    def synthesize_grasp(self, object_shape, grasp_type='cylindrical'):
        """
        Synthesize grasp for multi-fingered hand
        """
        if grasp_type == 'cylindrical':
            return self.cylindrical_grasp(object_shape)
        elif grasp_type == 'spherical':
            return self.spherical_grasp(object_shape)
        elif grasp_type == 'lateral':
            return self.lateral_grasp(object_shape)
        else:
            return self.calculate_optimal_grasp(object_shape)

    def cylindrical_grasp(self, object_shape):
        """
        Generate cylindrical grasp for objects like bottles, cups
        """
        # Find optimal finger positions around object circumference
        contact_points = self.calculate_cylindrical_contacts(object_shape)

        # Calculate finger angles and forces
        grasp_configuration = {
            'finger_positions': contact_points,
            'finger_forces': self.calculate_cylindrical_forces(contact_points),
            'hand_configuration': self.calculate_hand_configuration(contact_points)
        }

        return grasp_configuration

    def calculate_grasp_matrix(self, contact_points, surface_normals):
        """
        Calculate grasp matrix for force analysis
        """
        # Grasp matrix G relates joint torques to contact forces
        # G = [J_c^T, J_o^T]^T where J_c is contact Jacobian
        G = np.zeros((6 + len(contact_points) * 3, len(contact_points) * 3))

        for i, (point, normal) in enumerate(zip(contact_points, surface_normals)):
            # Contact Jacobian for point i
            J_c_i = np.zeros((6, 3))  # 6 DOF task space, 3 DOF contact force
            J_c_i[:3, :] = np.eye(3)  # Linear forces
            J_c_i[3:, :] = self.skew_symmetric(point)  # Torque components

            # Insert into grasp matrix
            start_idx = i * 3
            G[6:, start_idx:start_idx+3] = J_c_i

        return G

    def check_force_closure(self, grasp_matrix):
        """
        Check if grasp provides force closure
        """
        # Force closure exists if grasp can resist any external wrench
        # This involves checking if the grasp matrix spans the wrench space
        U, s, Vh = np.linalg.svd(grasp_matrix[6:])  # Consider only force part

        # Check if we have sufficient rank
        rank = np.sum(s > 1e-6)
        return rank >= 6  # Need to span 6D wrench space
```

## Grasp Stability Analysis

### Contact Stability Metrics

```python
class GraspStabilityAnalyzer:
    def __init__(self):
        self.friction_coefficient = 0.5

    def analyze_grasp_stability(self, contact_points, object_mass, external_forces=None):
        """
        Analyze stability of grasp under external disturbances
        """
        if external_forces is None:
            external_forces = np.array([0, 0, -9.81 * object_mass])  # Gravity

        # Calculate grasp wrench space
        wrench_space = self.calculate_grasp_wrench_space(contact_points)

        # Calculate required wrench to resist external forces
        required_wrench = self.calculate_required_wrench(external_forces)

        # Check if required wrench is within grasp capability
        is_stable = self.is_wrench_in_cone(required_wrench, wrench_space)

        # Calculate stability margin
        stability_margin = self.calculate_stability_margin(
            required_wrench, wrench_space
        )

        return {
            'is_stable': is_stable,
            'stability_margin': stability_margin,
            'wrench_space': wrench_space,
            'required_wrench': required_wrench
        }

    def calculate_grasp_wrench_space(self, contact_points):
        """
        Calculate the wrench space that the grasp can generate
        """
        # For each contact point, calculate the set of possible wrenches
        # given friction constraints
        wrench_cones = []

        for point in contact_points:
            # Calculate friction cone for this contact
            friction_cone = self.calculate_friction_cone(point)
            wrench_cones.append(friction_cone)

        # Combine all wrench cones to get total grasp capability
        total_wrench_space = self.combine_wrench_cones(wrench_cones)

        return total_wrench_space

    def calculate_friction_cone(self, contact_point):
        """
        Calculate friction cone at a contact point
        """
        # Friction cone: |tangent_force| <= mu * normal_force
        mu = self.friction_coefficient
        cone_matrix = np.array([
            [1, 0, 0, 0, 0, 0],  # Normal force constraint
            [0, 1, 0, 0, 0, 0],  # Tangent force 1
            [0, 0, 1, 0, 0, 0],  # Tangent force 2
        ])

        # Apply friction constraints
        # |F_tangent| <= mu * F_normal
        return cone_matrix
```

## Learning-based Manipulation

### Reinforcement Learning for Grasping

```python
import torch
import torch.nn as nn
import numpy as np

class LearningBasedGrasping(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.grasp_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Sigmoid()  # Output grasp success probability
        )

    def forward(self, state):
        encoded = self.state_encoder(state)
        grasp_success_prob = self.grasp_predictor(encoded)
        return grasp_success_prob

class RLGraspPlanner:
    def __init__(self):
        self.model = LearningBasedGrasping(state_dim=100, action_dim=1)
        self.model.load_state_dict(torch.load('grasp_model.pth'))
        self.model.eval()

    def predict_grasp_success(self, object_features, robot_state):
        """
        Predict probability of grasp success using learned model
        """
        # Combine object features and robot state
        state_vector = torch.cat([
            torch.tensor(object_features, dtype=torch.float32),
            torch.tensor(robot_state, dtype=torch.float32)
        ])

        with torch.no_grad():
            success_prob = self.model(state_vector)

        return success_prob.item()

    def plan_learning_based_grasp(self, object_observation):
        """
        Plan grasp using learned model and optimization
        """
        # Generate candidate grasps
        candidate_grasps = self.generate_candidate_grasps(object_observation)

        # Evaluate each grasp with learned model
        best_grasp = None
        best_score = -1

        for grasp in candidate_grasps:
            score = self.predict_grasp_success(
                self.extract_features(grasp, object_observation),
                self.get_robot_state()
            )

            if score > best_score:
                best_score = score
                best_grasp = grasp

        return best_grasp, best_score
```

## Safety and Compliance

### Collision Avoidance

```python
class ManipulationSafetySystem:
    def __init__(self):
        self.collision_checker = CollisionChecker()
        self.velocity_limiter = VelocityLimiter()
        self.force_monitor = ForceMonitor()

    def safe_execute_trajectory(self, trajectory, environment_map):
        """
        Execute trajectory with safety monitoring
        """
        for waypoint in trajectory:
            # Check collision at waypoint
            if self.collision_checker.check_collision(waypoint, environment_map):
                raise CollisionException("Trajectory would cause collision")

            # Monitor execution
            current_state = self.robot.get_state()
            if not self.is_safe_state(current_state, waypoint):
                self.emergency_stop()
                raise UnsafeStateException("Unsafe condition detected")

        # Execute trajectory with velocity limits
        limited_trajectory = self.velocity_limiter.apply_limits(trajectory)
        self.robot.execute_trajectory(limited_trajectory)

    def is_safe_state(self, current_state, target_state):
        """
        Check if current state is safe relative to target
        """
        # Check joint limits
        if not self.check_joint_limits(current_state):
            return False

        # Check velocity limits
        if not self.check_velocity_limits(current_state):
            return False

        # Check force limits
        if not self.check_force_limits(current_state):
            return False

        # Check proximity to obstacles
        if not self.check_obstacle_proximity(current_state):
            return False

        return True

    def emergency_stop(self):
        """
        Execute emergency stop procedure
        """
        # Gradually reduce velocities
        self.robot.set_joint_velocities(np.zeros(self.robot.num_joints))

        # Open gripper to release object if holding
        self.robot.open_gripper()

        # Switch to compliant control mode
        self.robot.set_control_mode('compliant')
```

## Best Practices

### Manipulation Design Guidelines

1. **Grasp Planning**: Always consider multiple grasp options and their robustness
2. **Force Control**: Implement force feedback for delicate operations
3. **Collision Avoidance**: Plan trajectories that avoid obstacles
4. **Compliance**: Use compliant control for contact-rich tasks
5. **Sensing**: Integrate multiple sensors for robust manipulation
6. **Safety**: Implement comprehensive safety checks and emergency procedures
7. **Calibration**: Regularly calibrate grippers and tool frames
8. **Testing**: Test extensively with various objects and conditions

### Implementation Tips

- Use appropriate coordinate frames for consistent transformations
- Implement proper error handling for grasp failures
- Consider object properties (mass, fragility, shape) in planning
- Use simulation for testing before real-world deployment
- Implement progressive learning from successful/unsuccessful grasps
- Plan for graceful degradation when sensors fail
- Document grasp parameters and their effects on success rates