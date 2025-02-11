import os
import numpy as np
import pinocchio as pin
from robot_properties_go1.go1_wrapper import Go1Robot, Go1Config
from envs.pybullet_env import PyBulletEnv
import torch
import pandas as pd 
from torch.utils.tensorboard import SummaryWriter
import time 

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the robot model and its initial configuration
pin_robot = Go1Config.buildRobotWrapper()
q0 = np.array(Go1Config.initial_configuration)
q0[0:2] = 0.0  # Set initial x, y positions to 0
v0 = pin.utils.zero(pin_robot.model.nv)

# Create the PyBullet environment with the Go1 robot
robot = PyBulletEnv(Go1Robot, q0, v0)

# Load evaluation data from CSV file
data_path = "/home/anudeep/devel/workspace/src/data/go1_eval/go1_trot_eval_data_v2.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data path {data_path} does not exist.")
data = pd.read_csv(data_path).drop(columns=['time'])

# Extract relevant data columns

q_ref = data[['qj_' + str(i) for i in range(1, 13)]].values
v_ref = data[['dqj_' + str(i) for i in range(1, 13)]].values

# Last 12 columns are torques
actual_torques = data.iloc[:, -12:].values

taus_path = "/home/anudeep/devel/workspace/src/data/predicted_tau/predicted_torques.csv"
if not os.path.exists(taus_path):
    raise FileNotFoundError(f"Data path {taus_path} does not exist.")
taus_data = pd.read_csv(taus_path)

predicted_torques = taus_data[['tau_' + str(i) for i in range(1, 13)]].values

# print(len(predicted_torques))

# PD control gains
kp = np.array([24.0] * 12)  # 18
kd = np.array([3.0] * 12)   # 3.2

# Lists to store PD torques and predicted torques
total_tau_pd = []
combined_tau = []

# Ensure the tensorboard logging directory exists
log_dir = '/home/anudeep/devel/workspace/runs/go1_end2ned_tau_policy_taus'
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

# robot.start_recording('go1_end2ned_tau_policy.mp4')
# Loop through each reference state (ensure it doesn't exceed available data)
num_iterations = min(len(predicted_torques), len(q_ref))  # Use the smaller length
for o in range(num_iterations):
    # Get the current state of the robot
    qj, dqj = robot.get_state()
    q_current = np.array(qj[7:])  # Current joint positions
    v_current = np.array(dqj[6:])  # Current joint velocities
    
    # Compute PD control torques
    pos_err = q_ref[o] - q_current
    vel_err = v_ref[o] - v_current

    tau_pd = kp * pos_err + kd * vel_err

    # Append PD torques to the list
    total_tau_pd.append(tau_pd)

    # Combine PD torques and predicted torques
    fin_tau = predicted_torques[o] + tau_pd

    # Append the combined torques to the list
    combined_tau.append(fin_tau)

    # Send the final torques to the robot
    robot.send_joint_command(fin_tau)

    time.sleep(0.001) # Sleep for 1ms, i.e., 1000Hz

# robot.stop_recording()

# # Log the data for each joint torque after the loop
for o in range(num_iterations):
    for i in range(12):  # there are 12 joints/tau values to log
        writer.add_scalars(f'Joint {i+1}', {
            'Actual Torque': actual_torques[o, i],  
            'Predicted Torque': predicted_torques[o, i],
            'Combined Torque': combined_tau[o][i],
            'PD Torque': total_tau_pd[o][i]
        }, o)

# Close the writer after all iterations are complete 
writer.close()

print("Tensorboard logging completed successfully.")
