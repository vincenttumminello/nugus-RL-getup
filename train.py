from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from environment import GetUpEnv

# Create environment
env = GetUpEnv("nugus/nugus.xml")

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
model.learn(total_timesteps=1_000)

# Save the trained model
model.save("getup_robot")