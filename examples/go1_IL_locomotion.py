# main.py

import time
import numpy as np
import torch
import pinocchio as pin
from robot_properties_go1.go1_wrapper import Go1Robot, Go1Config
from envs.pybullet_env import PyBulletEnv
from go1_IL_model_utils import load_model_and_normalizer, preprocess_eval_data, predict_joint_positions

# === User Configurable Parameters ===
model_path = '/home/anudeep/devel/workspace/data_humanoids/models/stl_humaoinds_delta_trot_2560n.pth'
data_path = "/home/anudeep/devel/workspace/data_humanoids/trot/go1_trot_data_actions_eval.csv"
eval_steps = 3000       # Number of offline predictions to loop through
sim_duration = 10000    # Number of simulation steps at 1kHz

# === Load Model and Preprocess Offline Data ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, X_min, X_max, Y_min, Y_max = load_model_and_normalizer(model_path, device)
X_tensor = preprocess_eval_data(data_path, eval_steps, X_min, X_max).to(device)
predicted_qs = predict_joint_positions(model, X_tensor, Y_min, Y_max)  # [eval_steps, 12]

print(f"Loaded model and prepared {eval_steps} cyclic joint targets")

# === Initialize Robot Simulation ===
pin_robot = Go1Config.buildRobotWrapper()
q0 = np.array(Go1Config.initial_configuration)
q0[0:2] = 0.0
v0 = pin.utils.zero(pin_robot.model.nv)
robot = PyBulletEnv(Go1Robot, q0, v0)

# === PD Controller Gains ===
kp = np.array([65.0] * 12)
kd = np.array([5.0] * 12)

# === Run Cyclic Playback Loop ===
print(f"Running simulation for {sim_duration} steps...")
for step in range(sim_duration):
    qj, dqj = robot.get_state()
    q_current = np.array(qj[-12:])
    v_current = np.array(dqj[-12:])

    pred_q = predicted_qs[step % eval_steps]  # Loop through offline-predicted joint angles
    pos_err = pred_q - q_current
    vel_err = -v_current
    tau = kp * pos_err + kd * vel_err

    robot.send_joint_command(tau)
    time.sleep(0.001)

print(" Simulation complete.")
