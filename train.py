from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
from environment import GetUpEnv
import mujoco 

class DataLoggingCallback(BaseCallback):
    """Log extra data every step to the SB3 logger (visible in TensorBoard)."""
    def __init__(self, log_dir="./logs/CustomData/", verbose=0):
        super().__init__(verbose)
        self.writer = SummaryWriter(log_dir)

    def _unwrap_env(self, wrapped_env):
        # If this is a VecEnv, take the first sub-env
        env_list = getattr(wrapped_env, "envs", None)
        env0 = env_list[0] if env_list else wrapped_env

        # Prefer gym's unwrapped when available
        if hasattr(env0, "unwrapped"):
            try:
                return env0.unwrapped
            except Exception:
                pass

        # Fallback: follow .env links until base env
        while hasattr(env0, "env"):
            env0 = env0.env
        return env0

    def _on_step(self) -> bool:
        try:
            env0 = self._unwrap_env(self.training_env)

            # Read torso height from env's mujoco data
            torso_id = mujoco.mj_name2id(env0.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
            torso_height = float(env0.data.xpos[torso_id][2])

            # Read left foot height
            left_foot_id = mujoco.mj_name2id(env0.model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
            left_foot_height = float(env0.data.xpos[left_foot_id][2])

            # Read shoulder joint control signal
            left_shoulder_pitch_id = mujoco.mj_name2id(env0.model, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_pitch")
            left_shoulder_pitch_control = float(env0.data.ctrl[left_shoulder_pitch_id])
            right_shoulder_pitch_id = mujoco.mj_name2id(env0.model, mujoco.mjtObj.mjOBJ_JOINT, "right_shoulder_pitch")
            right_shoulder_pitch_control = float(env0.data.ctrl[right_shoulder_pitch_id])

            # Record to SB3 logger (will be picked up by TensorBoard)
            self.writer.add_scalar("custom/torso_height", torso_height, self.num_timesteps)
            self.writer.add_scalar("custom/left_shoulder_pitch_control", left_shoulder_pitch_control, self.num_timesteps)
            self.writer.add_scalar("custom/right_shoulder_pitch_control", right_shoulder_pitch_control, self.num_timesteps)
            self.writer.add_scalar("custom/left_foot_height", left_foot_height, self.num_timesteps)
            # self.logger.record("custom/torso_height", torso_height)
            # self.logger.record("custom/left_shoulder_pitch_control", left_shoulder_pitch_control)
            # self.logger.record("custom/right_shoulder_pitch_control", right_shoulder_pitch_control)
            # self.logger.record("custom/left_foot_height", left_foot_height)
        except Exception as e:
            if self.verbose:
                print("DataLoggingCallback error:", e)
        return True

    def _on_training_end(self) -> None:
        try:
            self.writer.flush()
            self.writer.close()
        except Exception:
            pass

# Create environment
env = GetUpEnv("nugus/scene.xml")

# Verify environment is correct
check_env(env)

# Create RL agent (PPO is good for continuous control)
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    tensorboard_log="./logs/"
)

# Train the agent
model.learn(total_timesteps=1_000_000)

# Save the trained model
model.save("getup_robot")

