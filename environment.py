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
        
        self.max_steps = 1000
        self.current_step = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset to initial lying down pose
        mujoco.mj_resetData(self.model, self.data)
        
        # Set lying face down position
        self.data.qpos[0] = 0.0  # x position
        self.data.qpos[1] = 0.0  # y position
        self.data.qpos[2] = 0.15  # z position (slightly above ground)
        
        # Set orientation face down (90° pitch forward)
        self.data.qpos[3] = 0.7071  # qw
        self.data.qpos[4] = 0.0     # qx
        self.data.qpos[5] = 0.7071  # qy
        self.data.qpos[6] = 0.0     # qz
        
        # Optional: randomize joint positions slightly (but not the torso pose)
        if options and options.get('randomize', False):
            self.data.qpos[7:] += np.random.uniform(-0.1, 0.1, self.model.nq - 7)
        
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
        # Apply action (scale to appropriate range)
        self.data.ctrl[:] = action * self.model.actuator_ctrlrange[:, 1]
        
        # Step the simulation
        mujoco.mj_step(self.model, self.data)
        
        # Get new observation
        obs = self._get_obs()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        self.current_step += 1
        terminated = bool(self._is_standing()) # Success condition
        truncated = self.current_step >= self.max_steps
        
        return obs, reward, terminated, truncated, {}
    
    def _calculate_reward(self):
        # Reward based on torso height and uprightness
        # You'll need to adjust body names to match your robot
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
        torso_height = self.data.xpos[torso_id][2]  # z-coordinate
        
        # Reward for being upright
        height_reward = torso_height
        
        # Penalty for excessive control effort
        ctrl_cost = 0.01 * np.sum(np.square(self.data.ctrl))
        
        # Bonus for being upright (vertical orientation)
        upright_reward = 0.0
        if torso_height > 0.5:  # Adjust threshold for your robot
            torso_quat = self.data.xquat[torso_id]
            # Check if torso is vertical (depends on your robot's orientation)
            upright_reward = 2.0
        
        return height_reward + upright_reward - ctrl_cost
    
    def _is_standing(self):
        # Define success: torso is above certain height and stable
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
        torso_height = self.data.xpos[torso_id][2]
        
        # Check if standing and relatively stable
        is_high_enough = torso_height > 0.8  # Adjust for your robot
        is_stable = np.abs(self.data.qvel).mean() < 0.5
        
        return is_high_enough and is_stable