import time
import numpy as np
import pinocchio as pin
import torch
import pandas as pd
import matplotlib.pyplot as plt

from robot_properties_go1.go1_wrapper import Go1Robot, Go1Config
from mpc.go1_cyclic_gen import Go1MpcGaitGen
from motions.cyclic.go1_motion import trot
from envs.pybullet_env import PyBulletEnv
from controllers.robot_id_controller import InverseDynamicsController
from END2ENDPredictor import NMPCPredictor  # Your custom model class

# ────────── Load Model ──────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = '/home/anudeep/devel/workspace/data_humanoids/models/stl_humaoinds_trot_512n_phase.pth'

checkpoint = torch.load(MODEL_PATH, map_location=device)
X_min = checkpoint['x_min'].values if hasattr(checkpoint['x_min'], 'values') else checkpoint['x_min']
X_max = checkpoint['x_max'].values if hasattr(checkpoint['x_max'], 'values') else checkpoint['x_max']
Y_min = checkpoint['y_min'].values if hasattr(checkpoint['y_min'], 'values') else checkpoint['y_min']
Y_max = checkpoint['y_max'].values if hasattr(checkpoint['y_max'], 'values') else checkpoint['y_max']

input_size = 31
output_size = 12
neurons = 512

model = NMPCPredictor(input_size, output_size, neurons=neurons).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ────────── Model Parameters ──────────
MODEL_FREQ = 100  # Hz
model_dt = 1.0 / MODEL_FREQ
model_steps = 0
last_model_time = 0.0
q_des_model = np.zeros(12)
q_des_prev_model = np.zeros(12)

# Gains for swing/stance phases

# kp_swing = np.array([30.0, 30.0, 30.0] * 4)
# kd_swing = np.array([1.0, 1.0, 1.0] * 4)
# kp_stance = np.array([30.0, 30.0, 30.0] * 4)
# kd_stance = np.array([1.0, 1.0, 1.0] * 4)

kp_current = np.array([
    45.0, 15.0, 15.0,  # Leg 1: hip, thigh, knee
    45.0, 15.0, 20.0,  # Leg 2: hip, thigh, knee  
    45.0, 15.0, 20.0,  # Leg 3: hip, thigh, knee
    45.0, 15.0, 15.0   # Leg 4: hip, thigh, knee
])
kd_current = np.array([
    4.0, 1.5, 2.75,  # Leg 1: hip, thigh, knee
    4.0, 1.5, 1.5,  # Leg 2: hip, thigh, knee
    4.0, 1.5, 2.75,  # Leg 3: hip, thigh, knee
    4.0, 1.5, 1.5   # Leg 4: hip, thigh, knee
])


# ─────────────── 1. Robot & MPC Setup ───────────────
pin_robot = Go1Config.buildRobotWrapper()
urdf_path = Go1Config.urdf_path

rmodel = Go1Config().robot_model
rdata = rmodel.createData()

n_eff = 4
q0 = np.array(Go1Config.initial_configuration)
q0[0:2] = 0.0

v0 = pin.utils.zero(pin_robot.model.nv)
x0 = np.concatenate([q0, pin.utils.zero(pin_robot.model.nv)])

f_arr = ["FL_foot_fixed", "FR_foot_fixed", "RL_foot_fixed", "RR_foot_fixed"] 

v_des = np.array([0.3, 0.0, 0.0])  
w_des = 0.0
plan_freq = 0.05
update_time = 0.0

sim_t = 0.0
sim_dt = 0.001
index = 0
pln_ctr = 0

gait_params = trot
gait_period = gait_params.gait_period

kp = gait_params.kp
kv = gait_params.kd

lag = int(update_time/sim_dt)
gg = Go1MpcGaitGen(pin_robot, urdf_path, x0, plan_freq, q0, None)

gg.update_gait_params(gait_params, sim_t)

robot = PyBulletEnv(Go1Robot, q0, v0)

robot_id_ctrl = InverseDynamicsController(pin_robot, f_arr)
robot_id_ctrl.set_gains(gait_params.kp, gait_params.kd)

trj = 5 * 1000
simulation_time = trj + 1

state_id = robot.saveState()
num_failure = 0
q_des_prev = np.zeros(12)

# Logging setup
log_data = []

def calculate_tau(q_des, q_current, dq_current, kp, kd):
    pos_err = q_des - q_current
    vel_err = -dq_current
    return kp * pos_err + kd * vel_err

# Timing control for real-time simulation
start_time = time.perf_counter()
next_control_time = start_time + sim_dt

# ─────────────── 2. Control Loop ───────────────
for o in range(simulation_time):    
    # Real-time synchronization
    current_time = time.perf_counter()
    while current_time < next_control_time:
        current_time = time.perf_counter()
    next_control_time += sim_dt
    
    q, v = robot.get_state()
    
    # ────────── MPC Planning (20Hz) ──────────
    if pln_ctr == 0:
        contact_configuration = robot.get_current_contacts()
        xs_plan, us_plan, f_plan = gg.optimize(q, v, np.round(sim_t,3), v_des, w_des)

    if o < int(plan_freq/sim_dt) - 1:
        xs = xs_plan
        us = us_plan
        f = f_plan

    elif pln_ctr == lag and o > int(plan_freq/sim_dt)-1:
        lag = 0
        xs = xs_plan[lag:]
        us = us_plan[lag:]
        f = f_plan[lag:]
        index = 0
    
    # ────────── Model Inference (100Hz) ──────────
    if sim_t - last_model_time >= model_dt:
        # Get current state for model input
        qj_current = q[7:19]
        dqj_current = v[6:18]
        imu_gyro, imu_acc, _, _ = robot.get_imu_data()
        imu = np.concatenate([imu_acc, imu_gyro])
        phase = (sim_t % gait_period) / gait_period

        # Prepare input
        obs_raw = np.concatenate([imu, qj_current, dqj_current])
        obs_norm = 2 * (obs_raw - X_min) / (X_max - X_min) - 1
        model_input = np.concatenate([obs_norm, [phase]])

        # Run model
        with torch.no_grad():
            input_tensor = torch.tensor(model_input, dtype=torch.float32).unsqueeze(0).to(device)
            norm_action = model(input_tensor).cpu().numpy().squeeze()

        # Denormalize and smooth
        q_des_raw = 0.5 * (norm_action + 1) * (Y_max - Y_min) + Y_min
        alpha = 0.3
        q_des_model = alpha * q_des_raw + (1 - alpha) * q_des_prev_model
        q_des_prev_model = q_des_model
        last_model_time = sim_t
    
    # ────────── Torque Calculation ──────────
    # Get current joint state
    qj_current = q[7:19]
    dqj_current = v[6:18]
    
    # Calculate phase for gain selection
    phase = (sim_t % gait_period) / gait_period
    
    # if phase < 0.5:  # Swing phase
    #     kp_current = kp_swing
    #     kd_current = kd_swing
    # else:            # Stance phase
    #     kp_current = kp_stance
    #     kd_current = kd_stance
    
    
    # Calculate model-based torques
    tau_model = calculate_tau(q_des_model, qj_current, dqj_current, kp_current, kd_current)
    
    # Calculate MPC-based torques
    tau_mpc = robot_id_ctrl.id_joint_torques(
        q, v, 
        xs[index][:pin_robot.model.nq].copy(), 
        xs[index][pin_robot.model.nq:].copy(), 
        us[index], f[index], contact_configuration
    )
    
    # ────────── Send Command ──────────
    if o < 3000:
        tau = tau_mpc
    else:
        tau = tau_model
    
    # tau = tau_mpc
    
    robot.send_joint_command(tau)
    
    # ────────── Logging ──────────
    log_entry = {
        'time': sim_t,
        'phase': phase,
        'q_des_model': q_des_model,
        'tau_model': tau_model,
        'tau_mpc': tau_mpc,
        'tau_actual': tau,
        'q': q,
        'v': v
    }
    log_data.append(log_entry)
    
    # ────────── Update Counters ──────────
    sim_t += sim_dt
    pln_ctr = int((pln_ctr + 1) % (plan_freq/sim_dt))
    index += 1

    # Reset if robot falls
    if (q[0] > 50 or q[0] < -50 or q[1] > 50 or q[1] < -50 or q[2] > 0.7 or q[2] < 0.1):
        robot.restoreState(state_id)
        v_des = np.array([0.0, 0.0, 0.0])
        num_failure += 1

print("num of failure =", num_failure)

# Save log data to a csv file 
log_file = "/home/anudeep/devel/workspace/data_humanoids/linference_logs/inference_logs_hybrid.csv"

# Create DataFrame with properly flattened data
columns = ['time', 'phase'] + \
          [f'q_des_model_{i+1}' for i in range(12)] + \
          [f'tau_model_{i+1}' for i in range(12)] + \
          [f'tau_mpc_{i+1}' for i in range(12)] + \
          [f'tau_actual_{i+1}' for i in range(12)] + \
          [f'q_{i+1}' for i in range(19)] + \
          [f'v_{i+1}' for i in range(18)]

# Build DataFrame row by row
rows = []
for entry in log_data:
    row = [
        entry['time'],
        entry['phase']
    ]
    row.extend(entry['q_des_model'])
    row.extend(entry['tau_model'])
    row.extend(entry['tau_mpc'])
    row.extend(entry['tau_actual'])
    row.extend(entry['q'])
    row.extend(entry['v'])
    rows.append(row)

# Create and save DataFrame
log_df = pd.DataFrame(rows, columns=columns)
log_df.to_csv(log_file, index=False)
print(f"[INFO] Log saved to {log_file}")

# ─────────────── 3. Plot Comparison of Torques ───────────────
print("Generating torque comparison plots...")

# Extract time vector
time_vec = log_df['time'].values

# Create figure with 12 subplots (4x3 grid)
plt.figure(figsize=(20, 15))
plt.suptitle('Torque Comparison: Model Prediction vs MPC', fontsize=16)

# Joint names for labeling
joint_names = [
    'FL_hip', 'FL_thigh', 'FL_calf',
    'FR_hip', 'FR_thigh', 'FR_calf',
    'RL_hip', 'RL_thigh', 'RL_calf',
    'RR_hip', 'RR_thigh', 'RR_calf'
]

for i in range(12):
    plt.subplot(4, 3, i+1)
    
    # Extract data for this joint
    model_tau = log_df[f'tau_model_{i+1}']
    mpc_tau = log_df[f'tau_mpc_{i+1}']
    
    # Plot both torque signals
    plt.plot(time_vec, model_tau, 'b-', linewidth=1.5, label='Model Prediction')
    plt.plot(time_vec, mpc_tau, 'r-', linewidth=1.0, alpha=0.7, label='MPC')
    
    # Format plot
    plt.title(f'{joint_names[i]} Joint')
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (Nm)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Set y-axis limits based on data range
    y_min = min(min(model_tau), min(mpc_tau))
    y_max = max(max(model_tau), max(mpc_tau))
    margin = 0.1 * (y_max - y_min)
    plt.ylim(y_min - margin, y_max + margin)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle

# Save and show plot
plot_file = log_file.replace('.csv', '_torque_comparison.png')
plt.savefig(plot_file, dpi=150)
print(f"[INFO] Torque comparison plot saved to {plot_file}")
plt.show()