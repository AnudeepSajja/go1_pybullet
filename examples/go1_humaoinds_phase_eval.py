import os
import time
import numpy as np
import torch
import pandas as pd
import pinocchio as pin
from END2ENDPredictor import NMPCPredictor
from robot_properties_go1.go1_wrapper import Go1Config, Go1Robot
from envs.pybullet_env import PyBulletEnv

# ─────────────────────────────────────────────────────────────
# 1) MODEL AND NORMALIZATION SETUP
# ─────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = '/home/anudeep/devel/workspace/data_humanoids/models/stl_humaoinds_trot_256n.pth'
checkpoint = torch.load(model_path, map_location=device)

X_min = checkpoint['x_min']
X_max = checkpoint['x_max']
Y_min = checkpoint['y_min']
Y_max = checkpoint['y_max']

input_size = checkpoint['model_state_dict']['hidden1.weight'].shape[1]
output_size = checkpoint['model_state_dict']['output.weight'].shape[0]
neurons = 256

model = NMPCPredictor(input_size=input_size, output_size=output_size, neurons=neurons).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# PD Gains
kp = np.array([75.0, 75.0, 75.0] * 4)
kd = np.array([5.0, 5.0, 5.0] * 4)

# Desired velocity
v_des = np.array([0.33, 0.0, 0.0])  # [vx, vy, yaw_rate]

# ─────────────────────────────────────────────────────────────
# 2) ROBOT INITIALIZATION
# ─────────────────────────────────────────────────────────────
pin_robot = Go1Config.buildRobotWrapper()
q0 = np.array(Go1Config.initial_configuration)
q0[:2] = 0.0
v0 = pin.utils.zero(pin_robot.model.nv)
env = PyBulletEnv(Go1Robot, q0, v0)

# ─────────────────────────────────────────────────────────────
# 3) MAIN INFERENCE + CONTROL LOOP
# ─────────────────────────────────────────────────────────────
log_data = []
phase = 0.0
phase_increment = 1/150
iteration = 0  # <── iteration counter

try:
    while True:
        # === Sensor readings ===
        q, dq = env.get_state()
        imu_gyro, imu_acc, _, _ = env.get_imu_data()
        contact = env.get_current_contacts()

        # === Construct observation with external gait phase ===
        raw_obs = np.hstack([
            imu_acc, imu_gyro,
            q[7:], dq[6:],
            v_des, contact,
            [phase]  # Append externally defined gait phase
        ])

        # Normalize first 33 features
        obs_norm = 2 * (raw_obs[:-8] - X_min) / (X_max - X_min) - 1
        processed_obs = np.concatenate([obs_norm, raw_obs[-8:]])  # v_des + contact + phase
        obs_tensor = torch.from_numpy(processed_obs).float().to(device).unsqueeze(0)

        # === Neural Network Inference ===
        with torch.no_grad():
            norm_action = model(obs_tensor).cpu().numpy().squeeze()
        denorm_action = ((norm_action + 1) / 2) * (Y_max - Y_min) + Y_min

        # === PD Controller ===
        joint_pos = np.array(q[7:])
        joint_vel = np.array(dq[6:])
        position_error = denorm_action - joint_pos
        velocity_error = -joint_vel
        tau = kp * position_error + kd * velocity_error

        # === Debug print ===
        print(f"Iter: {iteration:03d} | Phase: {phase:.2f} | PosErr: {np.linalg.norm(position_error):.3f} | TauNorm: {np.linalg.norm(tau):.3f}")

        # === Send to robot ===
        env.send_joint_command(tau)

        # === Log data ===
        log_data.append(np.concatenate([denorm_action, joint_pos, joint_vel, tau, [phase]]))

        # === Update phase ===
        if iteration >= 100:
            phase = (phase + phase_increment) % 1.0  # Cycle phase between 0 and 1
        else:
            phase = 0.0  # Hold at 0 initially

        iteration += 1
        time.sleep(1.0 / 100)

except KeyboardInterrupt:
    print("\n[INFO] Control loop interrupted by user.")

finally:
    # Save inference log
    log_file_path = "/home/anudeep/devel/workspace/data_humanoids/linference_logs/inference_log_vdes_with_phase.csv"
    columns = ['imu_acc_x', 'imu_acc_y', 'imu_acc_z',
               'imu_gyro_x', 'imu_gyro_y', 'imu_gyro_z'] + \
              [f"pred_joint_{i+1}" for i in range(12)] + \
              [f"curr_joint_{i+1}" for i in range(12)] + \
              [f"velocity_{i+1}" for i in range(12)] + \
              [f"tau_{i+1}" for i in range(12)] + ["gait_phase"]
    log_df = pd.DataFrame(log_data, columns=columns)
    log_df.to_csv(log_file_path, index=False)
    print(f"[INFO] Robot disconnected. Log saved to {log_file_path}")
