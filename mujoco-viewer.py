import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("nugus/scene.xml")
data = mujoco.MjData(model)

# Reset and set lying face down position
mujoco.mj_resetData(model, data)

# Set face down pose
data.qpos[0:7] = [0.0, 0.0, -0.15, 0.7071, 0.0, 0.7071, 0.0]
data.qvel[:] = 0

mujoco.mj_forward(model, data)

print(f"Starting height: {data.qpos[2]:.3f}m")
print(f"Torso orientation (quat): {data.qpos[3:7]}")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()