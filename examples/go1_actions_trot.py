import os
import time
import numpy as np
import torch
import pandas as pd
import pinocchio as pin
from torch.utils.tensorboard import SummaryWriter
from END2ENDPredictor import NMPCPredictor
from robot_properties_go1.go1_wrapper import Go1Robot, Go1Config
from envs.pybullet_env import PyBulletEnv

# === Configuration ===
model_path = '/home/anudeep/devel/workspace/data_humanoids/models/stl_humaoinds_trot_512n.pth'
data_path = "/home/anudeep/devel/workspace/data_humanoids/trot/go1_trot_data_actions_eval.csv"
# log_dir = '/home/anudeep/devel/workspace/runs/go1_combined_trot_eval'

# === Device ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Load Model Checkpoint ===
checkpoint = torch.load(model_path, map_location=device)

input_size = checkpoint['model_state_dict']['hidden1.weight'].shape[1]
output_size = checkpoint['model_state_dict']['output.weight'].shape[0]
neurons = 512

estimator = NMPCPredictor(input_size=input_size, output_size=output_size, neurons=neurons).to(device)
estimator.load_state_dict(checkpoint['model_state_dict'])
estimator.eval()

X_min = np.array(checkpoint['x_min'])  # Should be length 30
X_max = np.array(checkpoint['x_max'])
Y_min = np.array(checkpoint['y_min'])
Y_max = np.array(checkpoint['y_max'])

print("Model and normalization params loaded")

# === Load Evaluation Data (first 1000 rows) ===
df = pd.read_csv(data_path).head(5000)

# Extract features
imu_cols = [col for col in df.columns if col.startswith('imu_acc_') or col.startswith('imu_gyro_')]
qj_cols = [col for col in df.columns if col.startswith('qj_')]
dqj_cols = [col for col in df.columns if col.startswith('dqj_')]
foot_cols = [col for col in df.columns if col.startswith('foot_')]

X_all = df[imu_cols + qj_cols + dqj_cols + foot_cols]
assert X_all.shape[1] == 34, f"Expected 34 features, got {X_all.shape[1]}"

# Normalize only the first 30 features
X_to_normalize = X_all[imu_cols + qj_cols + dqj_cols]
X_foot = X_all[foot_cols]

assert len(X_min) == 30 and len(X_max) == 30
X_norm = 2 * (X_to_normalize - X_min) / (X_max - X_min) - 1
X_input = pd.concat([X_norm, X_foot], axis=1)

X_tensor = torch.tensor(X_input.values, dtype=torch.float32).to(device)

# === Predict Joint Positions ===
with torch.no_grad():
    predicted_qs_norm = estimator(X_tensor)

predicted_qs = ((predicted_qs_norm.cpu().numpy() + 1) / 2) * (Y_max - Y_min) + Y_min

# === Initialize Robot Environment ===
pin_robot = Go1Config.buildRobotWrapper()
q0 = np.array(Go1Config.initial_configuration)
q0[0:2] = 0.0
v0 = pin.utils.zero(pin_robot.model.nv)
robot = PyBulletEnv(Go1Robot, q0, v0)

# === Get Actual Torques ===
actual_torques = df[[f'tau_{i}' for i in range(1, 13)]].values

# === PD Controller Gains ===
kp = np.array([65.0] * 12)
kd = np.array([5.0] * 12)

# === Start Recording and TensorBoard Logging ===
# robot.start_recording('go1_combined_trot_eval.mp4')
# os.makedirs(log_dir, exist_ok=True)
# writer = SummaryWriter(log_dir)

predicted_tau = []

# === Control Loop ===
for t in range(len(predicted_qs)):
    qj, dqj = robot.get_state()
    q_current = np.array(qj[7:])  # joint positions
    v_current = np.array(dqj[6:])  # joint velocities

    pos_err = predicted_qs[t] - q_current
    vel_err = -v_current
    tau = kp * pos_err + kd * vel_err

    robot.send_joint_command(tau)
    predicted_tau.append(tau)

    # TensorBoard logging
    # for i in range(12):
    #     writer.add_scalars(f'Joint {i+1}', {
    #         'Actual Torque': actual_torques[t, i],
    #         'Calculated Torque': tau[i]
    #     }, t)

    time.sleep(0.001)  # 1kHz

# robot.stop_recording()
# writer.close()

print("Evaluation and control complete.")
