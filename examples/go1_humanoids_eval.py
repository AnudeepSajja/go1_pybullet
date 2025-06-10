import os
import time
import numpy as np
import torch
import pandas as pd
import pinocchio as pin
from END2ENDPredictor import NMPCPredictor
from robot_properties_go1.go1_wrapper import Go1Config, Go1Robot
from motions.cyclic.go1_motion import trot
from envs.pybullet_env import PyBulletEnv


# ──────────────── 1. Model + Normalization Setup ────────────────
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


# ──────────────── 2. PD Gains and Gait Parameters ────────────────
kp = np.array([40.0, 40.0, 65.0] * 4)
kd = np.array([7.0, 7.0, 7.0] * 4)

GAIT_PARAMS = trot
GAIT_PERIOD = GAIT_PARAMS.gait_period  # e.g., 0.7s for trot

# Desired velocity (can be updated dynamically later)
# v_des = np.array([0.33, 0.11, 0.0])  # [vx, vy, yaw_rate]

# ──────────────── 3. Robot Initialization ────────────────
pin_robot = Go1Config.buildRobotWrapper()
q0 = np.array(Go1Config.initial_configuration)
q0[:2] = 0.0
v0 = pin.utils.zero(pin_robot.model.nv)
env = PyBulletEnv(Go1Robot, q0, v0)


# ──────────────── 4. Inference + Control Loop ────────────────

log_data = []
sim_t = 0.0
SIM_DT = 0.001

# ──────────────── Warm-up Phase ────────────────
# let the robot touch the ground 
print("[INFO] Warming up the robot...")
for _ in range(2):
    contact_flags = env.get_current_contacts()
    print(contact_flags)
    env.send_joint_command(np.ones(12))  # Send torques

    
try:
    while True:
            
        # === Sensor readings ===
        q, dq = env.get_state()
        imu_gyro, imu_acc, _, _ = env.get_imu_data()
        contact_raw = env.get_current_contacts()
        # print(f"[INFO] Contacts: {contact_raw}")
        contact_flags = np.array([int(c) for c in contact_raw])  # 4D binary
        
        # === Compute gait phase [0–1]
        phase = (sim_t % GAIT_PERIOD) / GAIT_PERIOD

        # === Build 31D input: imu(6) + qj(12) + dqj(12) + phase(1)
        imu = np.concatenate([imu_acc, imu_gyro])      # 6D
        qj = q[7:]                                      # 12D
        dqj = dq[6:]                                    # 12D
        obs_raw = np.concatenate([imu, qj, dqj])        # 30D

        obs_norm = 2 * (obs_raw - X_min.values) / (X_max.values - X_min.values) - 1
        
        model_input = np.concatenate([obs_norm, [phase]])  # 31D input
        # model_input = np.concatenate([obs_norm, contact_flags, [phase]])  # 35D
        
        
        input_tensor = torch.tensor(model_input, dtype=torch.float32).unsqueeze(0).to(device)


        # === Neural Network Inference ===
        with torch.no_grad():
            norm_action = model(input_tensor).cpu().numpy().squeeze()
        denorm_action = ((norm_action + 1) / 2) * (Y_max - Y_min) + Y_min  # Denormalize action

        # === PD Controller ===
        joint_pos = np.array(q[7:])
        joint_vel = np.array(dq[6:])
        
        position_error = denorm_action - joint_pos
        velocity_error = 0 -joint_vel
        tau = kp * position_error + kd * velocity_error

        # === Send to robot ===
        env.send_joint_command(tau)

        # === Log data ===
        log_data.append(np.concatenate([denorm_action, joint_pos, joint_vel, tau]))

        sim_t += SIM_DT
        time.sleep(SIM_DT)

except KeyboardInterrupt:
    print("\n[INFO] Control loop interrupted by user.")

finally:
    # Save inference log
    log_file_path = "inference_log_khadiv_pd.csv"
    columns = [f"pred_joint_{i+1}" for i in range(12)] + \
              [f"curr_joint_{i+1}" for i in range(12)] + \
              [f"velocity_{i+1}" for i in range(12)] + \
              [f"tau_{i+1}" for i in range(12)]

    log_df = pd.DataFrame(log_data, columns=columns)
    log_df.to_csv(log_file_path, index=False)
    print(f"[INFO] Log saved to {log_file_path}")
