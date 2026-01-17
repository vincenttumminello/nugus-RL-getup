import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco

class GetUpEnv(gym.Env):
    def __init__(self, xml_path, max_steps=1000):
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
        
        self.max_steps = max_steps
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

        # Slew-rate constraint
        self.slew_rate = 15.0
        self.prev_action = np.zeros(self.model.nu, dtype=np.float64)


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

        self.prev_action[:] = 0.0
        
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
        desired_full = np.zeros(self.model.nu)
        # Null action on all actuators until 165 timesteps
        if self.current_step > 165:
            desired_full[self.enabled_mask] = action

        # Scale into actuator control range
        desired_full = desired_full * self.model.actuator_ctrlrange[:, 1]
        # Apply slew-rate constraint
        dt = getattr(self.model, "opt", None)
        dt = self.model.opt.timestep if dt is not None else 0.002

        # Per actuator max delta this step
        if np.isscalar(self.slew_rate):
            max_delta = float(self.slew_rate) * dt
            delta = np.clip(desired_full - self.prev_action, -max_delta, max_delta)
        else:
            max_delta = self.slew_rate * dt
            delta = np.clip(desired_full - self.prev_action, -max_delta, max_delta)

        applied_ctrl = self.prev_action + delta
        self.data.ctrl[:] = applied_ctrl
        # Store for next step
        self.prev_action[:] = applied_ctrl

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

        # Stage based rewards for uprightness of torso (not considering height or feet contacts yet)
        torso_rot = self.data.xmat[torso_id]
        torso_roll = np.arctan2(torso_rot[7], torso_rot[8])  # rotation around x-axis
        torso_pitch = np.arctan2(torso_rot[6], np.sqrt(torso_rot[7]**2 + torso_rot[8]**2))  # rotation around y-axis

        # Reward for being upright
        upright_reward = 0.0
        if abs(torso_roll) < 0.1 and abs(torso_pitch) < 0.1:
            upright_reward = 100.0  # Full reward for being upright
        elif abs(torso_roll) < 0.1 or abs(torso_pitch) < 0.1:
            upright_reward = 7.0  # Points for being upright in one axis
        elif abs(torso_roll) < 0.2 and abs(torso_pitch) < 0.2:
            upright_reward = 50.0  # Partial reward for being close to upright
        elif abs(torso_roll) < 0.2 or abs(torso_pitch) < 0.2:
            upright_reward = 3.0   # Less points for being less upright in one axis
        elif abs(torso_roll) < 0.4 and abs(torso_pitch) < 0.4:
            upright_reward = 10.0  # Small reward for being somewhat upright
        elif abs(torso_roll) < 0.6 and abs(torso_pitch) < 0.6:
            upright_reward = 5.0  # Very small reward for being tilted
        else:
            upright_reward = -1.0  # Penalty for being far from upright

        
        return upright_reward
    
    def _is_standing(self):
        # Define success: torso is above certain height and stable
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
        torso_height = self.data.xpos[torso_id][2]
        
        # Check if standing and relatively stable
        is_high_enough = torso_height > 0.8  # Adjust for your robot
        is_stable = np.abs(self.data.qvel).mean() < 0.5
        
        return is_high_enough and is_stable