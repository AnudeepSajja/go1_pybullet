import os
import numpy as np
import pinocchio as pin
from robot_properties_go1.go1_wrapper import Go1Robot, Go1Config
from envs.pybullet_env import PyBulletEnv
import time
import torch
import pandas as pd
from motions.cyclic.go1_motion import trot
from END2ENDPredictor import NMPCPredictor  # Your custom model class

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATH = '/home/anudeep/devel/workspace/data_humanoids/models/stl_humaoinds_trot_512n_phase.pth'

# ────────── Load Model ──────────
checkpoint = torch.load(MODEL_PATH, map_location=device)
X_min, X_max = checkpoint['x_min'], checkpoint['x_max']
Y_min, Y_max = checkpoint['y_min'], checkpoint['y_max']

# Determine model architecture from checkpoint
input_size = 31
output_size = 12
neurons = 512

# Initialize model
model = NMPCPredictor(input_size, output_size, neurons=neurons).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

pin_robot = Go1Config.buildRobotWrapper()
q0 = np.array(Go1Config.initial_configuration)
q0[0:2] = 0.0
v0 = pin.utils.zero(pin_robot.model.nv)

robot = PyBulletEnv(Go1Robot, q0, v0)

control_loop_freq = 1000  # Hz
sim_dt = 1.0 / control_loop_freq

desired_freq = 100  # Hz
control_steps_per_update = int(control_loop_freq / desired_freq)

# kp_array = np.array([60.0, 60.0, 60.0] * 4)  # Position gains
# kd_array = np.array([5.0, 5.0, 5.0] * 4)     # Velocity gains

# kp_array = np.array([
#     65.0, 65.0, 65.0,  # Leg 1: hip, thigh, knee
#     65.0, 65.0, 65.0,  # Leg 2: hip, thigh, knee  
#     65.0, 65.0, 65.0,  # Leg 3: hip, thigh, knee
#     65.0, 65.0, 65.0,  # Leg 4: hip, thigh, knee
# ])

# kd_array = np.array([
#     5.0, 5.0, 5.0,  # Leg 1: hip, thigh, knee
#     5.0, 5.0, 5.0,  # Leg 2: hip, thigh, knee
#     5.0, 5.0, 5.0,  # Leg 3: hip, thigh, knee
#     5.0, 5.0, 5.0,   # Leg 4: hip, thigh, knee
# ])

# Gains for swing phase
kp_swing = np.array([40.0, 45.0, 45.0,
                     40.0, 45.0, 45.0,
                     40.0, 45.0, 45.0,
                     40.0, 45.0, 45.0])  # Position gains for swing phase

kd_swing = np.array([5.7, 5.5, 5.0,
                     5.7, 5.5, 5.0,
                     5.7, 5.5, 5.0,
                     5.7, 5.5, 5.0])  # Velocity gains for swing phase

# Gains for stance phase
kp_stance = np.array([40.0, 40.0, 40.0,
                      40.0, 40.0, 40.0,
                      40.0, 40.0, 40.0,
                      40.0, 40.0, 40.0])  # Position gains for stance phase

kd_stance = np.array([5.0, 5.0, 5.0,
                      5.0, 5.0, 5.0,
                      5.0, 5.0, 5.0,
                      5.0, 5.0, 5.0])  # Velocity gains for stance phase


gait_period = trot.gait_period
sim_t = 0.0

# Initialize logging
log_data = []

# Function to calculate tau
def calculate_tau(pred, kp, kd):
    qj_cur, dqj_cur = robot.get_state()
    qj_current = np.array(qj_cur[7:])  # Current joint positions (12)
    dqj_current = np.array(dqj_cur[6:])  # Current joint velocities (12)
    
    pos_err = pred - qj_current
    vel_err = np.zeros(12) - dqj_current 
    tau = kp * pos_err + kd * vel_err
    return tau

# Timing control for real-time simulation
start_time = time.perf_counter()
next_control_time = start_time + sim_dt

iterations = 5000  # Number of iterations to run the control loop

try:
    for o in range(iterations):  # Run for a fixed number of iterations or until a stop condition
        # Get current state
        qj, dqj = robot.get_state()
        q_current = np.array(qj[7:])  # Joint positions (12)
        v_current = np.array(dqj[6:])  # Joint velocities (12)
        
        # Simulate IMU data (replace with actual IMU data if available)
        imu_gyro, imu_acc, _, _ = robot.get_imu_data()
        imu = np.concatenate([imu_acc, imu_gyro])
        
        # Calculate phase
        phase = round( (sim_t % gait_period) / gait_period, 3)
        
        if phase < 0.5:
            # Swing phase
            kp = kp_swing
            kd = kd_swing
        else:
            # Stance phase
            kp = kp_stance
            kd = kd_stance
        
        # Prepare input data: IMU(6) + qj(12) + dqj(12) + phase(1) = 31
        obs_raw = np.concatenate([imu, q_current, v_current])
        
        # Normalize input
        obs_norm = 2 * (obs_raw - X_min.values) / (X_max.values - X_min.values) - 1
        model_input = np.concatenate([obs_norm, [phase]])
        
        # Run inference
        with torch.no_grad():
            input_tensor = torch.tensor(model_input, dtype=torch.float32).unsqueeze(0).to(device)
            norm_action = model(input_tensor).cpu().numpy().squeeze()
        
        # Denormalize action
        latest_q_pred = 0.5 * (norm_action + 1) * (Y_max.values - Y_min.values) + Y_min.values
        
        # Calculate torques
        tau = calculate_tau(latest_q_pred, kp, kd)
        
        # Log data
        log_data.append(np.concatenate([latest_q_pred, tau, [phase]]))
        
        print(f"Step: {o}, Phase: {phase:.3f}")
        
        # Send torque commands
        robot.send_joint_command(tau)
        
        # Real-time synchronization
        current_time = time.perf_counter()
        while current_time < next_control_time:
            current_time = time.perf_counter()
        next_control_time += sim_dt
        
        sim_t += sim_dt

finally:
    # Log all values correctly across time
    log_file = "/home/anudeep/devel/workspace/data_humanoids/linference_logs/inference_logs_online.csv"
    columns = [f"q_des_{i+1}" for i in range(12)] + \
              [f"tau_{i+1}" for i in range(12)] + \
              ["phase"]
    pd.DataFrame(log_data, columns=columns).to_csv(log_file, index=False)
    print(f"[INFO] Log saved to {log_file}")