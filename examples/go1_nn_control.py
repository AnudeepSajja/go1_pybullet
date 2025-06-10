import numpy as np
import pinocchio as pin
from robot_properties_go1.go1_wrapper import Go1Robot, Go1Config
from envs.pybullet_env import PyBulletEnv
import torch
import pandas as pd

from END2ENDPredictor import NMPCPredictor  


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model checkpoint
model_path = '/home/anudeep/devel/workspace/models/end2end_predictor_norm.pth'
checkpoint = torch.load(model_path, map_location=device)

# Initialize the model
input_size = checkpoint['model_state_dict']['hidden1.weight'].shape[1]
output_size = checkpoint['model_state_dict']['output.weight'].shape[0]
neurons = 2048  # or whatever value you used during training

# Initialize estimator with neurons parameter
estimator = NMPCPredictor(input_size=input_size, output_size=output_size, neurons=neurons).to(device)
estimator.load_state_dict(checkpoint['model_state_dict'])

# Load normalization parameters
X_min, X_max = checkpoint['x_min'], checkpoint['x_max']
Y_min, Y_max = checkpoint['y_min'], checkpoint['y_max']

print("Model loaded successfully along with normalization parameters.")

# Initialize the robot model and its initial configuration
pin_robot = Go1Config.buildRobotWrapper()
q0 = np.array(Go1Config.initial_configuration)
q0[0:2] = 0.0  # Set initial x, y positions to 0
v0 = pin.utils.zero(pin_robot.model.nv)

# Create the PyBullet environment with the Go1 robot
robot = PyBulletEnv(Go1Robot, q0, v0)

# Load evaluation data from CSV file
data_path = "/home/anudeep/devel/workspace/src/data/go1_eval/go1_trot_data_eval.csv" # works
# data_path = "/home/anudeep/devel/workspace/src/data/data_without_noise/go1_trot_data_eval_test.csv"
data = pd.read_csv(data_path).drop(columns=['time'])

# Extract relevant data columns
base_pos = data[['base_pos_x', 'base_pos_y', 'base_pos_z']].values
base_ori = data[['base_ori_x', 'base_ori_y', 'base_ori_z', 'base_ori_w']].values
base_vel = data[['base_vel_x', 'base_vel_y', 'base_vel_z']].values
imu_acc = data[['imu_acc_x', 'imu_acc_y', 'imu_acc_z']].values
imu_gyro = data[['imu_gyro_x', 'imu_gyro_y', 'imu_gyro_z']].values
foot_positions = [data[f'foot_{i}'].values for i in range(1, 5)]
q_ref = data[['qj_' + str(i) for i in range(1, 13)]].values
v_ref = data[['dqj_' + str(i) for i in range(1, 13)]].values

# last 12 columns are torques
actual_torques = data.iloc[:, -12:].values

# PD control gains
kp = np.array([18] * 12)
kd = np.array([3.2] * 12)


# Lists to store PD torques and predicted torques
total_tau_pd = []
total_predicted_torques = []
combined_tau = []

# Loop through each reference state
for o in range(10000):    
    # Get the current state of the robot
    qj, dqj = robot.get_state()
    q_current = np.array(qj[7:])  # Current joint positions
    v_current = np.array(dqj[6:])  # Current joint velocities
    
    # Compute PD control torques
    pos_err = q_ref[o+1] - q_current
    vel_err = v_ref[o+1] - v_current

    tau_pd = kp * pos_err + kd * vel_err

    # Append PD torques to the list
    total_tau_pd.append(tau_pd)


    # Prepare input data for the estimator
    X = np.hstack([
        base_pos[o], base_ori[o], base_vel[o],
        imu_acc[o], imu_gyro[o], q_ref[o], v_ref[o],
        foot_positions[0][o], foot_positions[1][o], foot_positions[2][o], foot_positions[3][o]
    ])

    # Normalize all but the last 4 columns
    X[:-4] = 2 * (X[:-4] - X_min) / (X_max - X_min) - 1

    # Drop the first two columns of X (i.e., base_pos_x and base_pos_y)
    X = X[2:]

    # Convert to tensor and move to device
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    # Predict torques using the estimator
    estimator.eval()
    with torch.no_grad():
        predicted_torques = estimator(X_tensor)

    predicted_torques_numpy = predicted_torques.cpu().numpy()  

    # Unnormalize the predicted torques
    predicted_torques_unnormalized = ((predicted_torques_numpy + 1) / 2) * (Y_max - Y_min) + Y_min
    
    total_predicted_torques.append(predicted_torques_unnormalized)

    # Combine PD torques and predicted torques
    fin_tau = predicted_torques_unnormalized

    fin_tau = np.fmin(np.fmax(fin_tau, -20), 20)

    # Append the combined torques to the list
    combined_tau.append(fin_tau)


    zero_torques = np.zeros(12)

    # Send the final torques to the robot
    robot.send_joint_command(fin_tau)

    print(f"Step {o + 1} completed")

# Convert lists to numpy arrays for further analysis
total_tau_pd_np = np.array(total_tau_pd)
total_predicted_torques_np = np.array(total_predicted_torques)
combined_tau = np.array(combined_tau)

# Print the shapes of the resulting arrays
print("Total tau_imp shape:", total_tau_pd_np.shape)
print("Total predicted torques shape:", total_predicted_torques_np.shape)



# plot the torques for each joint with actual and predicted values
import matplotlib.pyplot as plt

# Assuming you have 12 joints
num_joints = 12

# Create a figure with subplots for each joint
fig, axs = plt.subplots(num_joints, figsize=(12, 18), sharex=True)


# Plotting each joint's torques
for i in range(num_joints):
    axs[i].plot(actual_torques[:, i], label='Actual Torques', color='blue')
    axs[i].plot(total_predicted_torques_np[:, i], label='Predicted Torques', color='red')
    axs[i].plot(total_tau_pd_np[:, i], label='PD Torques', color='green')
    axs[i].plot(combined_tau[:, i], label='Combined Torques', color='orange')
    axs[i].set_title(f"Joint {i + 1}")
    axs[i].legend()
    axs[i].grid()
# Set common x-axis label
plt.xlabel('Time Steps')
plt.tight_layout()
plt.show()

