import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco

class GetUpEnv(gym.Env):
    def __init__(self, xml_path):
        super().__init__()
        
        # Load your robot model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Define action space (joint torques/positions)
        # Adjust size based on your robot's actuators
        n_actions = self.model.nu  # number of actuators
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(n_actions,), dtype=np.float32
        )
        
        # Define observation space (joint positions, velocities, etc.)
        n_obs = self.model.nq + self.model.nv  # positions + velocities
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )
        
        self.max_steps = 5000
        self.current_step = 0

        # Define which actuators to disable
        self.disabled_actuators = ["neck_yaw", "head_pitch"]
        self.disabled_indices = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) 
            for name in self.disabled_actuators
        ]

        # Create a mask for enabled actuators
        self.enabled_mask = np.ones(self.model.nu, dtype=bool)
        self.enabled_mask[self.disabled_indices] = False

        # Action space only for enabled actuators
        n_actions = self.enabled_mask.sum()
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(n_actions,), dtype=np.float32
        )

        # Needed joint addresses
        left_shoulder_pitch_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'left_shoulder_pitch')
        self.left_shoulder_pitch_vel_adr = self.model.jnt_dofadr[left_shoulder_pitch_id]
        right_shoulder_pitch_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'right_shoulder_pitch')
        self.right_shoulder_pitch_vel_adr = self.model.jnt_dofadr[right_shoulder_pitch_id]
        left_shoulder_roll_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'left_shoulder_roll')
        self.left_shoulder_roll_vel_adr = self.model.jnt_dofadr[left_shoulder_roll_id]
        right_shoulder_roll_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'right_shoulder_roll')
        self.right_shoulder_roll_vel_adr = self.model.jnt_dofadr[right_shoulder_roll_id]
        left_elbow_pitch_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'left_elbow_pitch')
        self.left_elbow_pitch_vel_adr = self.model.jnt_dofadr[left_elbow_pitch_id]
        right_elbow_pitch_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'right_elbow_pitch')
        self.right_elbow_pitch_vel_adr = self.model.jnt_dofadr[right_elbow_pitch_id]
        left_hip_yaw_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'left_hip_yaw')
        self.left_hip_yaw_vel_adr = self.model.jnt_dofadr[left_hip_yaw_id]
        right_hip_yaw_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'right_hip_yaw')
        self.right_hip_yaw_vel_adr = self.model.jnt_dofadr[right_hip_yaw_id]
        left_hip_roll_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'left_hip_roll')
        self.left_hip_roll_vel_adr = self.model.jnt_dofadr[left_hip_roll_id]
        right_hip_roll_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'right_hip_roll')
        self.right_hip_roll_vel_adr = self.model.jnt_dofadr[right_hip_roll_id]
        left_hip_pitch_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'left_hip_pitch')
        self.left_hip_pitch_vel_adr = self.model.jnt_dofadr[left_hip_pitch_id]
        right_hip_pitch_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'right_hip_pitch')
        self.right_hip_pitch_vel_adr = self.model.jnt_dofadr[right_hip_pitch_id]
        left_knee_pitch_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'left_knee_pitch')
        self.left_knee_pitch_vel_adr = self.model.jnt_dofadr[left_knee_pitch_id]
        right_knee_pitch_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'right_knee_pitch')
        self.right_knee_pitch_vel_adr = self.model.jnt_dofadr[right_knee_pitch_id]

        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset to initial lying down pose
        mujoco.mj_resetData(self.model, self.data)
        
        # Set lying face down position
        self.data.qpos[0] = 0.0  # x position
        self.data.qpos[1] = 0.0  # y position
        self.data.qpos[2] = 0.5  # z position (slightly above ground)
        
        # Set orientation face down (90° pitch forward)
        self.data.qpos[3] = 0.7071  # qw
        self.data.qpos[4] = 0.0     # qx
        self.data.qpos[5] = 0.7071  # qy
        self.data.qpos[6] = 0.0     # qz

        # Set shoulder pitch joints to negative 90 (by the side)
        left_should_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'left_shoulder_pitch')
        left_shoulder_adr = self.model.jnt_qposadr[left_should_id]
        right_should_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'right_shoulder_pitch')
        right_shoulder_adr = self.model.jnt_qposadr[right_should_id]
        self.data.qpos[left_shoulder_adr] = 0 # doesn't seem to do what I want it to do
        self.data.qpos[right_shoulder_adr] = 0 # not sure what's going on with these
        
        # TODO: randomize joint positions slightly (but not the torso pose)
        # self.data.qpos[7:] += np.random.uniform(-0.1, 0.1, self.model.nq - 7)
        
        # Zero all velocities
        self.data.qvel[:] = 0
        
        self.current_step = 0
        mujoco.mj_forward(self.model, self.data)
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        # Return current state (positions and velocities)
        return np.concatenate([
            self.data.qpos.flat.copy(),
            self.data.qvel.flat.copy(),
        ]).astype(np.float32)
    
    def step(self, action):
        # Map reduced action to full control vector
        full_ctrl = np.zeros(self.model.nu)
        # Null action on all actuators until 150 timesteps
        if self.current_step > 150:
            full_ctrl[self.enabled_mask] = action

        # Apply action (scale to appropriate range)
        self.data.ctrl[:] = full_ctrl * self.model.actuator_ctrlrange[:, 1]
        
        # Step the simulation
        mujoco.mj_step(self.model, self.data)
        
        # Get new observation
        obs = self._get_obs()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        self.current_step += 1
        # terminated = bool(self._is_standing()) # Success condition
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        return obs, reward, terminated, truncated, {}
    
    def _calculate_reward(self):
        # Reward based on torso height and uprightness
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
        torso_height = self.data.xpos[torso_id][2]  # z-coordinate

        # Reward for being upright
        height_reward = torso_height
        
        # Penalty for excessive control effort
        ctrl_cost = 0.1 * np.sum(np.square(self.data.ctrl))
        
        # Bonus for being upright (vertical orientation)
        upright_reward = 0.0
        if torso_height > 0.5  and torso_height < 1.0:  
            torso_rot = self.data.xmat[torso_id]
            # Check if torso is vertical (bottom row of rotation matrix close to [0, 0, 1])
            if abs(torso_rot[6]) < 1e-3 and abs(torso_rot[7]) < 1e-3 and abs(torso_rot[8] - 1) < 1e-3:
                upright_reward = 1000.0
            else:
                upright_reward = 50.0 # Partial reward for being close in height range
        elif torso_height >= 1.1:
            upright_reward = -200.0  # Penalize overshooting too high
        elif torso_height > 0.1 and torso_height <= 0.3:
            upright_reward = 10.0  # Small reward for being off the ground
        elif torso_height > 0.3 and torso_height <= 0.5:
            upright_reward = 20.0  # Larger reward for being mid-height
        else:
            upright_reward = -10.0  # Penalize being too low

        # Reward for having feet attached to ground (progressive)
        left_foot_force_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'left_foot_force')
        right_foot_force_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'right_foot_force')
    
        left_foot_force = np.linalg.norm(self.data.sensordata[left_foot_force_id:left_foot_force_id+3])
        right_foot_force = np.linalg.norm(self.data.sensordata[right_foot_force_id:right_foot_force_id+3])

        # Stage-based reward
        # Early stage: Reward getting feet under body and in contact
        # Late stage: Reward standing with weight on both feet
        
        if torso_height < 0.3:
            # Early: Just reward any foot contact
            foot_reward = 0.1 * (min(left_foot_force / 10.0, 1.0) + min(right_foot_force / 10.0, 1.0))
        elif torso_height < 0.6:
            # Mid: Reward balanced foot contact
            foot_reward = 1 * min(left_foot_force, right_foot_force) / 20.0
        else:
            # Standing: Reward strong, balanced support
            total_force = left_foot_force + right_foot_force
            balance = 1.0 - abs(left_foot_force - right_foot_force) / (total_force + 1e-6)
            foot_reward = 1.0 * (total_force / 50.0) * balance * 2

        ## Penalise high limb velocities 
        # Get joint velocities
        left_hip_pitch_vel = self.data.qvel[self.left_hip_pitch_vel_adr]
        right_hip_pitch_vel = self.data.qvel[self.right_hip_pitch_vel_adr]
        left_hip_roll_vel = self.data.qvel[self.left_hip_roll_vel_adr]
        right_hip_roll_vel = self.data.qvel[self.right_hip_roll_vel_adr]
        left_shoulder_pitch_vel = self.data.qvel[self.left_shoulder_pitch_vel_adr]
        right_shoulder_pitch_vel = self.data.qvel[self.right_shoulder_pitch_vel_adr]
        left_shoulder_roll_vel = self.data.qvel[self.left_shoulder_roll_vel_adr]
        right_shoulder_roll_vel = self.data.qvel[self.right_shoulder_roll_vel_adr]
        left_elbow_pitch_vel = self.data.qvel[self.left_elbow_pitch_vel_adr]
        right_elbow_pitch_vel = self.data.qvel[self.right_elbow_pitch_vel_adr]
        left_knee_pitch_vel = self.data.qvel[self.left_knee_pitch_vel_adr]
        right_knee_pitch_vel = self.data.qvel[self.right_knee_pitch_vel_adr]
        limb_velocities = np.array([
            left_hip_pitch_vel, right_hip_pitch_vel,
            left_hip_roll_vel, right_hip_roll_vel,
            left_shoulder_pitch_vel, right_shoulder_pitch_vel,
            left_shoulder_roll_vel, right_shoulder_roll_vel,
            left_knee_pitch_vel, right_knee_pitch_vel
        ])
        # Penalise joints moving too fast ( > 2pi rad/s)
        vel_penalty = 0.01 * np.sum(np.maximum(np.abs(limb_velocities) - 6.28, 0))

        
        return height_reward + upright_reward + foot_reward - ctrl_cost - vel_penalty
    
    def _is_standing(self):
        # Define success: torso is above certain height and stable
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
        torso_height = self.data.xpos[torso_id][2]
        
        # Check if standing and relatively stable
        is_high_enough = torso_height > 0.8  # Adjust for your robot
        is_stable = np.abs(self.data.qvel).mean() < 0.5
        
        return is_high_enough and is_stable