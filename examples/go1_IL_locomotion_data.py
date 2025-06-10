# go1_IL_locomotion.py


import torch
import time
import numpy as np
import pinocchio as pin
import random
from mim_data_utils import DataLogger
from robot_properties_go1.go1_wrapper import Go1Robot, Go1Config
from mpc.go1_cyclic_gen import Go1MpcGaitGen
from motions.cyclic.go1_motion import trot, jump, stand, bound
from envs.pybullet_terrain_env import PyBulletTerrainEnv
from controllers.robot_id_controller import InverseDynamicsController
import csv

from go1_IL_model_utils import load_model_and_normalizer, preprocess_eval_data, predict_joint_positions

# === User Configurable Parameters ===
model_path = '/home/anudeep/devel/workspace/data_humanoids/models/stl_humaoinds_delta_trot_2560n.pth'
data_path = "/home/anudeep/devel/workspace/data_humanoids/trot/go1_trot_data_actions_eval.csv"
output_csv_path = "/home/anudeep/devel/workspace/data_humanoids/trot/go1_IL_collected_data.csv"
eval_steps = 3000       # Number of offline predictions to loop through
sim_duration = 10000    # Number of simulation steps at 1kHz

# === Load Model and Preprocess Offline Data ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, X_min, X_max, Y_min, Y_max = load_model_and_normalizer(model_path, device)
X_tensor = preprocess_eval_data(data_path, eval_steps, X_min, X_max).to(device)
predicted_qs = predict_joint_positions(model, X_tensor, Y_min, Y_max)  # [eval_steps, 12]

print(f"✅ Loaded model and prepared {eval_steps} cyclic joint targets")

# === Initialize Robot Simulation ===
pin_robot = Go1Config.buildRobotWrapper()
q0 = np.array(Go1Config.initial_configuration)
q0[0:2] = 0.0
v0 = pin.utils.zero(pin_robot.model.nv)
robot = PyBulletTerrainEnv(Go1Robot, q0, v0)

# === PD Controller Gains ===
kp = np.array([65.0] * 12)
kd = np.array([5.0] * 12)

# === Run Cyclic Playback Loop with Data Logging ===
print(f"🚶 Running simulation for {sim_duration} steps...")
data_log = []

for step in range(sim_duration):
    if step % 100 == 0:
        robot.save_image(step)

    qj, dqj = robot.get_state()
    q_current = np.array(qj[-12:])
    v_current = np.array(dqj[-12:])

    pred_q = predicted_qs[step % eval_steps]
    pos_err = pred_q - q_current
    vel_err = -v_current
    tau = kp * pos_err + kd * vel_err
    robot.send_joint_command(tau)

    # === Sensor & State Logging ===
    imu_gyro, imu_acc, _, _ = robot.get_imu_data()
    foot_contact = robot.get_current_contacts()

    # Append a single row of relevant data
    row = list(imu_acc) + list(imu_gyro) + list(q_current) + list(v_current) + list(foot_contact) + list(pred_q)
    data_log.append(row)

    time.sleep(0.001)

# === Save Data to CSV ===
header = [
    "imu_acc_x", "imu_acc_y", "imu_acc_z",
    "imu_gyro_x", "imu_gyro_y", "imu_gyro_z",
] + [f"qj_{i+1}" for i in range(12)] + [f"dqj_{i+1}" for i in range(12)] + [f"foot_{i+1}" for i in range(4)] + [f"pred_q_{i+1}" for i in range(12)]

with open(output_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(data_log)

print(f"✅ Simulation complete. Data saved to {output_csv_path}")
