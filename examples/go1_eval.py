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

# from test_slider import PDGainController  

# # Init
# pd_slider = PDGainController()
# kp_infer = np.array(pd_slider.get_kp_array())
# kd_infer = np.array(pd_slider.get_kd_array())

# ─────────────── 1. Setup ───────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = '/home/anudeep/devel/workspace/data_humanoids/models/stl_humaoinds_trot_512n_phase.pth'
checkpoint = torch.load(model_path, map_location=device)
X_min, X_max = checkpoint['x_min'], checkpoint['x_max']
Y_min, Y_max = checkpoint['y_min'], checkpoint['y_max']

input_size = checkpoint['model_state_dict']['hidden1.weight'].shape[1]
output_size = checkpoint['model_state_dict']['output.weight'].shape[0]
model = NMPCPredictor(input_size, output_size, neurons=512).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ─────────────── 2. PD Gains ───────────────
kp_kick = np.array([60.0, 60.0, 60.0] * 4)
kd_kick = np.array([5.0, 5.0, 5.0] * 4)


# kp_infer = np.array([30.0, 50.0, 45.0] * 4)
# kd_infer = np.array([2.0, 2.0, 2.0] * 4)

# Define kp and kd values for 4 legs, each with 3 joints (hip, thigh, knee)
kp_infer = np.array([
    40.0, 40.0, 50.0,  # Leg 1: hip, thigh, knee
    40.0, 40.0, 50.0,  # Leg 2: hip, thigh, knee  
    40.0, 40.0, 50.0,  # Leg 3: hip, thigh, knee
    40.0, 40.0, 50.0,  # Leg 4: hip, thigh, knee
])

kd_infer = np.array([
    5.0, 5.0, 5.0,  # Leg 1: hip, thigh, knee
    5.0, 5.0, 5.0,  # Leg 2: hip, thigh, knee
    5.0, 5.0, 5.0,  # Leg 3: hip, thigh, knee
    5.0, 5.0, 5.0,   # Leg 4: hip, thigh, knee
])


# ─────────────── 3. Gait Params ───────────────
GAIT_PERIOD = trot.gait_period
v_des = np.array([0.3, 0.0, 0.0])  # [vx, vy, yaw_rate]

# ─────────────── 4. Env Init ───────────────
pin_robot = Go1Config.buildRobotWrapper()
q0 = np.array(Go1Config.initial_configuration)
q0[:2] = 0.0
v0 = pin.utils.zero(pin_robot.model.nv)
env = PyBulletEnv(Go1Robot, q0, v0)

# ─────────────── 5. Kickstart ───────────────
kick_df = pd.read_csv("/home/anudeep/devel/workspace/data_humanoids/predictions/humaoids_stl_trot_q_phase.csv")
kick_qs = kick_df[[f"a{i}" for i in range(1, 13)]].values
kick_max_steps = min(1, len(kick_qs))  # ~1 second

# ─────────────── 6. Control Loop ───────────────
SIM_DT = 0.001
PLAN_DT = 0.001
sim_t = 0.0
last_infer_time = -PLAN_DT
latest_q_pred = np.zeros(12)
log_data = []


try:
    while True:
        
        # kp_infer = np.array(pd_slider.get_kp_array())
        # kd_infer = np.array(pd_slider.get_kd_array())
        
        # Get current state
        q, dq = env.get_state()
        imu_gyro, imu_acc, _, _ = env.get_imu_data()
        contact_flags = np.array([int(c) for c in env.get_current_contacts()])
        phase = round((sim_t % GAIT_PERIOD) / GAIT_PERIOD, 3)

        qj = q[7:]
        dqj = dq[6:]
        
        

        if sim_t < kick_max_steps * SIM_DT:
            # ───── Kickstart phase ─────
            q_pred = kick_qs[int(sim_t / SIM_DT)]
            tau = kp_kick * (q_pred - qj) - kd_kick * dqj
            latest_q_pred = q_pred
        else:
            
            print("INFERENCE PHASE")
            # ───── Inference every PLAN_DT seconds ─────
            if sim_t - last_infer_time >= PLAN_DT:
                imu = np.concatenate([imu_acc, imu_gyro])  # 6D
                obs_raw = np.concatenate([imu, qj, dqj])   # 30D

                obs_norm = 2 * (obs_raw - X_min.values) / (X_max.values - X_min.values) - 1
                
                # model_input = np.concatenate([obs_norm, [phase], v_des])  # 34D
                model_input = np.concatenate([obs_norm, [phase]])  # 31D

                input_tensor = torch.tensor(model_input, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    norm_action = model(input_tensor).cpu().numpy().squeeze()
                latest_q_pred = ((norm_action + 1) / 2) * (Y_max - Y_min) + Y_min
                last_infer_time = sim_t

            tau = kp_infer * (latest_q_pred - qj) - kd_infer * dqj
            # tau = kp_kick * (latest_q_pred - qj) - kd_kick * dqj
            # tau = latest_q_pred
            
            # clip tau for each joints
            # Per-joint torque limits (update these values as needed)
            # tau_min = np.array([-8, -15, -0.2, 0, -10, -0.2, -8, -10, -0.2, -0.2, -10, -0.2])
            # tau_max = np.array([ 3,  5,  15,  5.5,  0.5,  20,  3,  5,  15,  5.5,  0.5,  18])

            # # Clip tau values per joint
            # tau = np.clip(tau, tau_min, tau_max)

            
            # time.sleep(0.005) 
            # tau = latest_q_pred

        env.send_joint_command(tau)
        print(f"[INFO] Phase: {phase}, Contacts: {contact_flags}")

        # log_data.append(np.concatenate([qj, dqj, tau, contact_flags, [phase], v_des]))
        log_data.append(np.concatenate([latest_q_pred, tau, [phase]]))

        sim_t += SIM_DT
        # time.sleep(SIM_DT)
        # time.sleep(1)  # Reduced sleep for faster simulation

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user.")

finally:
    # Log all values correctly across time
    log_file = "/home/anudeep/devel/workspace/data_humanoids/linference_logs/inference_log_kickstart_online.csv"
    columns = [f"q_des_{i+1}" for i in range(12)] + \
            [f"tau_{i+1}" for i in range(12)] + \
            ["phase"]
    pd.DataFrame(log_data, columns=columns).to_csv(log_file, index=False)
    print(f"[INFO] Log saved to {log_file}")

