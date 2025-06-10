import time
import numpy as np
import torch
import pandas as pd
import pinocchio as pin
from END2ENDPredictor import NMPCPredictor  # Your custom model class
from robot_properties_go1.go1_wrapper import Go1Config, Go1Robot
from envs.pybullet_env import PyBulletEnv

# ────────── Configuration ──────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = '/home/anudeep/devel/workspace/data_humanoids/models/stl_humaoinds_trot_512n_phase.pth'
SIM_DT = 0.01
PLAN_DT = 0.01  # 100 Hz planning frequency
GAIT_PERIOD = 0.5  # Example value - adjust to your actual gait period

# ────────── Load Model ──────────
checkpoint = torch.load(MODEL_PATH, map_location=device)
X_min, X_max = checkpoint['x_min'], checkpoint['x_max']
Y_min, Y_max = checkpoint['y_min'], checkpoint['y_max']

# Determine model architecture from checkpoint
input_size = checkpoint['model_state_dict']['hidden1.weight'].shape[1]
output_size = checkpoint['model_state_dict']['output.weight'].shape[0]
neurons = checkpoint['model_state_dict']['hidden1.weight'].shape[0]  # Number of neurons

# Initialize model
model = NMPCPredictor(input_size, output_size, neurons=neurons).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ────────── Robot Setup ──────────
kp = np.array([60.0, 60.0, 60.0] * 4)  # Position gains
kd = np.array([5.0, 5.0, 5.0] * 4)     # Velocity gains

pin_robot = Go1Config.buildRobotWrapper()
q0 = np.array(Go1Config.initial_configuration)
q0[:2] = 0.0
v0 = pin.utils.zero(pin_robot.model.nv)
env = PyBulletEnv(Go1Robot, q0, v0)

# ────────── Kickstart Setup ──────────
kick_df = pd.read_csv("/home/anudeep/devel/workspace/data_humanoids/predictions/humaoids_stl_trot_q_phase.csv")
kick_qs = kick_df[[f"a{i}" for i in range(1, 13)]].values
kick_max_steps = min(len(kick_qs), int(0.5 / SIM_DT))  # Max 0.5 seconds kickstart

# ────────── Control Loop ──────────
sim_t = 0.0
last_infer_time = -PLAN_DT
latest_q_pred = np.zeros(12)
log_data = []
inference_times = []

try:
    while True:
        # Get robot state
        q, dq = env.get_state()
        imu_gyro, imu_acc, _, _ = env.get_imu_data()
        qj = q[7:]
        dqj = dq[6:]
        
        # Calculate gait phase
        phase = (sim_t % GAIT_PERIOD) / GAIT_PERIOD

        # Kickstart phase (initial trajectory)
        if sim_t < kick_max_steps * SIM_DT:
            step_idx = int(sim_t / SIM_DT)
            q_pred = kick_qs[step_idx]
            tau = kp * (q_pred - qj) - kd * dqj
            latest_q_pred = q_pred
        
        # Main neural network control
        else:
            # Run inference at fixed frequency
            if sim_t - last_infer_time >= PLAN_DT:
                start_time = time.perf_counter()
                
                # Prepare input data: IMU(6) + qj(12) + dqj(12) + phase(1) = 31
                imu = np.concatenate([imu_acc, imu_gyro])
                obs_raw = np.concatenate([imu, qj, dqj])
                
                # Normalize input
                obs_norm = 2 * (obs_raw - X_min.values) / (X_max.values - X_min.values) - 1
                model_input = np.concatenate([obs_norm, [phase]])
                
                # Run inference
                with torch.no_grad():
                    input_tensor = torch.tensor(model_input, dtype=torch.float32).unsqueeze(0).to(device)
                    norm_action = model(input_tensor).cpu().numpy().squeeze()
                
                # Denormalize output to get desired joint positions
                latest_q_pred = ((norm_action + 1) / 2) * (Y_max - Y_min) + Y_min
                
                # Record inference time
                infer_time = (time.perf_counter() - start_time) * 1000
                inference_times.append(infer_time)
                print(f"Inference time: {infer_time:.2f}ms")
                last_infer_time = sim_t

            # Compute torques using latest prediction
            tau = kp * (latest_q_pred - qj) - kd * dqj

        # Send commands and log data
        env.send_joint_command(tau)
        log_data.append(np.concatenate([latest_q_pred, tau, [phase]]))
        
        # Update simulation time
        sim_t += SIM_DT
        time.sleep(SIM_DT)  # Uncomment for real-time simulation

except KeyboardInterrupt:
    print("\nSimulation stopped by user")

finally:
    # Save log data
    log_file = "/home/anudeep/devel/workspace/data_humanoids/linference_logs/inference_log.csv"
    columns = [f"q_des_{i+1}" for i in range(12)] + \
              [f"tau_{i+1}" for i in range(12)] + \
              ["phase"]
    pd.DataFrame(log_data, columns=columns).to_csv(log_file, index=False)
    
    # Print performance stats
    if inference_times:
        print(f"\nInference performance ({len(inference_times)} calls):")
        print(f"  Average: {np.mean(inference_times):.2f}ms")
        print(f"  Min: {np.min(inference_times):.2f}ms")
        print(f"  Max: {np.max(inference_times):.2f}ms")