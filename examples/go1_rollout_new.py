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
model_path = '/home/anudeep/devel/workspace/data_humanoids/models/stl_humaoinds_delta_trot_2560n.pth'
data_path = "/home/anudeep/devel/workspace/data_humanoids/trot/go1_trot_data_actions_eval.csv"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

eval_steps = 5000         # Number of offline steps to evaluate and loop over
sim_duration = 15000      # Total number of simulation steps (at 1kHz)

# === Load Model ===
checkpoint = torch.load(model_path, map_location=device)
input_size = checkpoint['model_state_dict']['hidden1.weight'].shape[1]
output_size = checkpoint['model_state_dict']['output.weight'].shape[0]
neurons = 2560

estimator = NMPCPredictor(input_size=input_size, output_size=output_size, neurons=neurons).to(device)
estimator.load_state_dict(checkpoint['model_state_dict'])
estimator.eval()

X_min = np.array(checkpoint['x_min'])  # shape (30,)
X_max = np.array(checkpoint['x_max'])
Y_min = np.array(checkpoint['y_min'])  # shape (12,)
Y_max = np.array(checkpoint['y_max'])

print("✅ Model and normalization parameters loaded.")

# === Load Evaluation Data ===
df = pd.read_csv(data_path).head(eval_steps)

imu_cols = [col for col in df.columns if col.startswith('imu_acc_') or col.startswith('imu_gyro_')]
qj_cols = [col for col in df.columns if col.startswith('qj_')]
dqj_cols = [col for col in df.columns if col.startswith('dqj_')]
foot_cols = [col for col in df.columns if col.startswith('foot_')]

X_all = df[imu_cols + qj_cols + dqj_cols + foot_cols]
assert X_all.shape[1] == 34, f"Expected 34 features, got {X_all.shape[1]}"

# === Normalize inputs (first 30 features only)
X_to_normalize = X_all[imu_cols + qj_cols + dqj_cols]
X_foot = X_all[foot_cols]
X_norm = 2 * (X_to_normalize - X_min) / (X_max - X_min + 1e-8) - 1
X_input = pd.concat([X_norm, X_foot], axis=1)

X_tensor = torch.tensor(X_input.values, dtype=torch.float32).to(device)

# === Predict Offline Joint Positions ===
with torch.no_grad():
    predicted_qs_norm = estimator(X_tensor)
predicted_qs = ((predicted_qs_norm.cpu().numpy() + 1) / 2) * (Y_max - Y_min) + Y_min  # shape: [eval_steps, 12]

print(f"✅ Predicted {eval_steps} joint position steps.")

# === Initialize Robot ===
pin_robot = Go1Config.buildRobotWrapper()
q0 = np.array(Go1Config.initial_configuration)
q0[0:2] = 0.0
v0 = pin.utils.zero(pin_robot.model.nv)
robot = PyBulletEnv(Go1Robot, q0, v0)

# === PD Gains ===
kp = np.array([65.0] * 12)
kd = np.array([5.0] * 12)

# === Cyclic Playback ===
print(f"🚶 Starting cyclic playback for {sim_duration} steps...")
for step in range(sim_duration):
    qj, dqj = robot.get_state()
    q_current = qj[-12:]
    v_current = dqj[-12:]

    # Loop over predicted_qs using modulo
    pred_q = predicted_qs[step % eval_steps]

    pos_err = pred_q - q_current
    vel_err = -v_current
    tau = kp * pos_err + kd * vel_err

    robot.send_joint_command(tau)
    time.sleep(0.001)  # Run at 1kHz

print("✅ Cyclic control loop finished.")
