import os
import time
import numpy as np
import torch
import pandas as pd
import pinocchio as pin
from END2ENDPredictor import NMPCPredictor
from robot_properties_go1.go1_wrapper import Go1Robot, Go1Config
from envs.pybullet_env import PyBulletEnv

# === Configuration ===
model_path = '/home/anudeep/devel/workspace/data_humanoids/models/stl_humaoinds_trot_512n.pth'
data_path = "/home/anudeep/devel/workspace/data_humanoids/trot/go1_trot_data_actions_eval.csv"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
steps_per_phase = 5000
sim_dt = 0.001

# === Load Model ===
checkpoint = torch.load(model_path, map_location=device)
input_size = checkpoint['model_state_dict']['hidden1.weight'].shape[1]
output_size = checkpoint['model_state_dict']['output.weight'].shape[0]
neurons = 512

estimator = NMPCPredictor(input_size=input_size, output_size=output_size, neurons=neurons).to(device)
estimator.load_state_dict(checkpoint['model_state_dict'])
estimator.eval()

X_min = np.array(checkpoint['x_min'])  # length 30
X_max = np.array(checkpoint['x_max'])
Y_min = np.array(checkpoint['y_min'])
Y_max = np.array(checkpoint['y_max'])

print("Model and normalization params loaded")

# === Load Eval Data for Phase 1 ===
df = pd.read_csv(data_path).head(steps_per_phase)

imu_cols = [col for col in df.columns if col.startswith('imu_acc_') or col.startswith('imu_gyro_')]
qj_cols = [col for col in df.columns if col.startswith('qj_')]
dqj_cols = [col for col in df.columns if col.startswith('dqj_')]
foot_cols = [col for col in df.columns if col.startswith('foot_')]

X_all = df[imu_cols + qj_cols + dqj_cols + foot_cols]
X_to_normalize = X_all[imu_cols + qj_cols + dqj_cols]
X_foot = X_all[foot_cols]

X_norm = 2 * (X_to_normalize - X_min) / (X_max - X_min) - 1
X_input = pd.concat([X_norm, X_foot], axis=1)
X_tensor = torch.tensor(X_input.values, dtype=torch.float32).to(device)

# === Predict Joint Positions for Phase 1 ===
with torch.no_grad():
    predicted_qs_norm = estimator(X_tensor)
predicted_qs = ((predicted_qs_norm.cpu().numpy() + 1) / 2) * (Y_max - Y_min) + Y_min

# === Initialize Robot ===
pin_robot = Go1Config.buildRobotWrapper()
q0 = np.array(Go1Config.initial_configuration)
q0[0:2] = 0.0
v0 = pin.utils.zero(pin_robot.model.nv)
robot = PyBulletEnv(Go1Robot, q0, v0)

# === PD Gains ===
kp = np.array([65.0] * 12)
kd = np.array([5.0] * 12)

# === Phase 1: Run with Evaluation Predictions & Collect Observations ===
print("Phase 1: Running with eval predictions and collecting observations...")
collected_obs = []

for t in range(steps_per_phase):
    qj, dqj = robot.get_state()
    q_current = np.array(qj[7:])
    v_current = np.array(dqj[6:])
    
    # Control
    pos_err = predicted_qs[t] - q_current
    vel_err = -v_current
    tau = kp * pos_err + kd * vel_err
    robot.send_joint_command(tau)

    # Collect observation from robot
    imu_gyro, imu_acc, _, _ = robot.get_imu_data()
    foot_contact = robot.get_current_contacts()

    obs_vec = np.concatenate([
        imu_acc, imu_gyro,
        np.array(qj[7:]),     # joint pos
        np.array(dqj[6:]),    # joint vel
        np.array(foot_contact)
    ])
    collected_obs.append(obs_vec)

    time.sleep(sim_dt)

# === Phase 2: Run using collected observations ===
print("Phase 2: Running with collected observations...")

X_collected = np.stack(collected_obs)
X_norm_phase2 = 2 * (X_collected[:, :30] - X_min) / (X_max - X_min) - 1
X_input_phase2 = np.concatenate([X_norm_phase2, X_collected[:, 30:]], axis=1)

X_tensor_phase2 = torch.tensor(X_input_phase2, dtype=torch.float32).to(device)
with torch.no_grad():
    predicted_qs_norm_phase2 = estimator(X_tensor_phase2)
predicted_qs_phase2 = ((predicted_qs_norm_phase2.cpu().numpy() + 1) / 2) * (Y_max - Y_min) + Y_min

for t in range(steps_per_phase):
    qj, dqj = robot.get_state()
    q_current = np.array(qj[7:])
    v_current = np.array(dqj[6:])

    # Control
    pos_err = predicted_qs_phase2[t] - q_current
    vel_err = -v_current
    tau = kp * pos_err + kd * vel_err
    robot.send_joint_command(tau)

    time.sleep(sim_dt)

print("Both rollout phases complete.")
