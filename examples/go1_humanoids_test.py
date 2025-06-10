import os
import time
import threading
import numpy as np
import torch
import pandas as pd
import pinocchio as pin
from END2ENDPredictor import NMPCPredictor
from robot_properties_go1.go1_wrapper import Go1Robot, Go1Config
from motions.cyclic.go1_motion import trot
from envs.pybullet_env import PyBulletEnv
from controllers.robot_id_controller import InverseDynamicsController
from mpc.go1_cyclic_gen import Go1MpcGaitGen

# ──────────────── 1. Load Neural Model ────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = '/home/anudeep/devel/workspace/data_humanoids/models/stl_humaoinds_trot_256n.pth'
checkpoint = torch.load(model_path, map_location=device)

X_min, X_max = checkpoint['x_min'].values, checkpoint['x_max'].values
Y_min, Y_max = checkpoint['y_min'].values, checkpoint['y_max'].values

input_size = checkpoint['model_state_dict']['hidden1.weight'].shape[1]
output_size = checkpoint['model_state_dict']['output.weight'].shape[0]
model = NMPCPredictor(input_size=input_size, output_size=output_size, neurons=256).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ──────────────── 2. Configurations ────────────────
GAIT_PARAMS = trot
GAIT_PERIOD = GAIT_PARAMS.gait_period
kp = np.array([45.0, 65.0, 70.0] * 4)
kd = np.array([6.0, 6.0, 6.0] * 4)
SIM_DT = 0.001
INFER_DT = 0.02  # 50 Hz
SWITCH_STEP = 1000

# ──────────────── 3. Robot Initialization ────────────────
pin_robot = Go1Config.buildRobotWrapper()
q0 = np.array(Go1Config.initial_configuration)
q0[:2] = 0.0
v0 = pin.utils.zero(pin_robot.model.nv)
env = PyBulletEnv(Go1Robot, q0, v0)

# ──────────────── 4. NMPC Setup ────────────────
urdf_path = Go1Config.urdf_path
x0 = np.concatenate([q0, v0])
gg = Go1MpcGaitGen(pin_robot, urdf_path, x0, 0.05, q0, None)
gg.update_gait_params(GAIT_PARAMS, 0.0)

id_ctrl = InverseDynamicsController(pin_robot, [
    "FL_foot_fixed", "FR_foot_fixed", "RL_foot_fixed", "RR_foot_fixed"
])
id_ctrl.set_gains(GAIT_PARAMS.kp, GAIT_PARAMS.kd)

# ──────────────── 5. Shared Data ────────────────
q_des_shared = np.zeros(12)
lock = threading.Lock()
sim_t = 0.0
step_counter = 0
log_data = []

# ──────────────── 6. Model Inference Thread ────────────────
def model_inference_loop():
    global sim_t
    while True:
        if step_counter < SWITCH_STEP:
            time.sleep(INFER_DT)
            continue

        q, dq = env.get_state()
        imu_gyro, imu_acc, _, _ = env.get_imu_data()
        joint_pos = q[7:]
        joint_vel = dq[6:]
        phase = (sim_t % GAIT_PERIOD) / GAIT_PERIOD

        imu = np.concatenate([imu_acc, imu_gyro])
        obs_raw = np.concatenate([imu, joint_pos, joint_vel])
        obs_norm = 2 * (obs_raw - X_min) / (X_max - X_min) - 1
        model_input = np.concatenate([obs_norm, [phase]])
        input_tensor = torch.tensor(model_input, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            norm_action = model(input_tensor).cpu().numpy().squeeze()
        q_des = ((norm_action + 1) / 2) * (Y_max - Y_min) + Y_min

        with lock:
            q_des_shared[:] = q_des

        time.sleep(INFER_DT)

# ──────────────── 7. Control Loop (1 kHz) ────────────────
print("[INFO] Starting hybrid control loop...")

try:
    # Start model thread
    t_model = threading.Thread(target=model_inference_loop)
    t_model.start()

    while True:
        q, dq = env.get_state()
        joint_pos = q[7:]
        joint_vel = dq[6:]
        phase = (sim_t % GAIT_PERIOD) / GAIT_PERIOD

        if step_counter < SWITCH_STEP:
            if step_counter % int(0.05 / SIM_DT) == 0:  # Every 20 Hz
                try:
                    contact_config = [bool(c) for c in env.get_current_contacts()]
                    xs_plan, us_plan, f_plan = gg.optimize(q, dq, round(sim_t, 3), np.array([0.2, 0.04, 0.0]), 0.0)
                    q_plan = xs_plan[0][:pin_robot.model.nq]
                    v_plan = xs_plan[0][pin_robot.model.nq:]
                    tau = id_ctrl.id_joint_torques(q, dq, q_plan, v_plan, us_plan[0], f_plan[0], contact_config)
                    q_des = joint_pos + (tau + kd * joint_vel) / kp
                    with lock:
                        q_des_shared[:] = q_des
                except Exception as e:
                    print(f"[ERROR] NMPC failed: {e}")
                    break

        with lock:
            q_des = q_des_shared.copy()

        # PD control
        position_error = q_des - joint_pos
        velocity_error = -joint_vel
        tau = kp * position_error + kd * velocity_error
        env.send_joint_command(tau)

        log_data.append(np.concatenate([q_des, joint_pos, joint_vel, tau, [int(step_counter >= SWITCH_STEP)]]))

        time.sleep(SIM_DT)
        sim_t += SIM_DT
        step_counter += 1

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user.")

finally:
    columns = [f"q_des_{i}" for i in range(12)] + \
              [f"q_curr_{i}" for i in range(12)] + \
              [f"dq_curr_{i}" for i in range(12)] + \
              [f"tau_{i}" for i in range(12)] + ["is_nn"]
    df = pd.DataFrame(log_data, columns=columns)
    df.to_csv("hybrid_control_log.csv", index=False)
    print("[INFO] Log saved to hybrid_control_log.csv")
