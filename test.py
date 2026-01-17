from environment import GetUpEnv
from stable_baselines3 import SAC

# Load trained model
model = SAC.load("getup_robot")

# Test with visualization
env = GetUpEnv("nugus/scene.xml")
obs, _ = env.reset()

import mujoco.viewer

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        viewer.sync()
        
        if terminated or truncated:
            obs, _ = env.reset()