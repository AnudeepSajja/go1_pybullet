import os
import numpy as np
import pinocchio as pin
from robot_properties_go1.go1_wrapper import Go1Robot, Go1Config
from envs.pybullet_env import PyBulletEnv
import torch
import pandas as pd
from END2ENDPredictor import NMPCPredictor  
from matplotlib import pyplot as plt
import time 

from torch.utils.tensorboard import SummaryWriter

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

data_path = "/home/anudeep/devel/workspace/src/data/jump_data/go1_jump_data_eval.csv"
# data_path = "/home/anudeep/devel/workspace/src/data/go1_eval/go1_trot_data_eval.csv"

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data path {data_path} does not exist.")
data = pd.read_csv(data_path).drop(columns=['time'])


# Extract the actual torques
actual_torques = data[['tau_' + str(i) for i in range(1, 13)]].values

# Load the predicted Qs
# 
qs_path = "/home/anudeep/devel/workspace/src/data/predicted_tau/predicted_action_q_jump.csv"
# qs_path = "/home/anudeep/devel/workspace/src/data/predicted_tau/predicted_action_q_trot.csv"

if not os.path.exists(qs_path):
    raise FileNotFoundError(f"Data path {qs_path} does not exist.")
qs_path = pd.read_csv(qs_path)

predicted_qs = qs_path[['a' + str(i) for i in range(1, 13)]].values

# PD control gains
kp = np.array([75.0] * 12)  # Experiment with lower values

# kd for each joint
kd = np.array([5] * 12)  # Experiment with lower values


# Initialize the list to store the combined torques
predicted_tau = []

# num_iterations = 5000

# Ensure the number of iterations does not exceed the available predicted data
num_iterations = len(predicted_qs)

for o in range(num_iterations):
    # get the current joint positions and velocities
    qj, dqj = robot.get_state()
    q_current = np.array(qj[7:])  # Current joint positions
    v_current = np.array(dqj[6:])  # Current joint velocities

    # Compute the torques
    pos_err = predicted_qs[o] - q_current
    vel_err = np.zeros(12) - v_current

    tau_cal = kp * pos_err + kd * vel_err

    # Send the torques to the robot
    robot.send_joint_command(tau_cal)

    # Append the torques to the list
    predicted_tau.append(tau_cal)

    time.sleep(0.001) # Sleep for 1ms, i.e., 1000Hz


# Ensure the tensorboard logging directory exists
log_dir = '/home/anudeep/devel/workspace/runs/go1_nn_control_trot'
# log_dir = '/home/anudeep/devel/workspace/runs/go1_nn_control_jump'
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)


# Log the data for each joint torque after the loop
for o in range(num_iterations):
    for i in range(12):  # there are 12 joints/tau values to log
        writer.add_scalars(f'Joint {i+1}', {
            'Actual Torque': actual_torques[o, i],  
            'Predicted Torque': predicted_tau[o][i]
        }, o)

# plot the actual torques and the predicted torques
actual_torques = np.array(actual_torques)
predicted_tau = np.array(predicted_tau)
plt.figure()
plt.plot(actual_torques[:, 0], label='Actual Torque')
plt.plot(predicted_tau[:, 0], label='Predicted Torque')
plt.legend()
plt.show()



