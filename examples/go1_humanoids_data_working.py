import time
import numpy as np
import pinocchio as pin
import csv
from numpy.linalg import pinv
from robot_properties_go1.go1_wrapper import Go1Robot, Go1Config
from mpc.go1_cyclic_gen import Go1MpcGaitGen
from motions.cyclic.go1_motion import trot
from envs.pybullet_env import PyBulletEnv
from controllers.robot_id_controller import InverseDynamicsController
from sys_identification import SystemIdentification
import os

# ======= Configuration =======
PIN_ROBOT = Go1Config.buildRobotWrapper()
URDF_PATH = Go1Config.urdf_path
ROBOT_CFG = Go1Config()

# System identification setup
SYSID = SystemIdentification(
    URDF_PATH,
    config_file="/home/anudeep/devel/workspace/src/robot_properties_go1/src/robot_properties_go1/resources/go1_config.yaml",
    floating_base=True
)

# Simulation parameters
SIM_DT = 0.001
PLAN_FREQ = 0.05  # MPC replanning frequency (20 Hz)
TRAJ_LEN = 3000    # 5 seconds per rollout
MAX_ROLLOUTS = 1
V_DES = np.array([0.3   , 0.0, 0.0])  # Desired velocity
W_DES = 0.0                         # Desired yaw rate

# Gait parameters
GAIT_PARAMS = trot
KP = GAIT_PARAMS.kp
KV = GAIT_PARAMS.kd

# Initialize robot and controllers
q0 = np.array(ROBOT_CFG.initial_configuration)
q0[0:2] = 0.0  # Reset base position
v0 = pin.utils.zero(PIN_ROBOT.model.nv)
x0 = np.concatenate([q0, v0])

# Initialize simulation
robot = PyBulletEnv(Go1Robot, q0, v0)
gg = Go1MpcGaitGen(PIN_ROBOT, URDF_PATH, x0, PLAN_FREQ, q0, None)
gg.update_gait_params(GAIT_PARAMS, 0.0)

# Inverse dynamics controller
ctrl = InverseDynamicsController(PIN_ROBOT, [
    "FL_foot_fixed", "FR_foot_fixed", "RL_foot_fixed", "RR_foot_fixed"
])
ctrl.set_gains(KP, KV)

# Data storage
dataset = []
header = [
    *[f"imu_acc_{i}" for i in range(3)],
    *[f"imu_gyro_{i}" for i in range(3)],
    *[f"qj_{i}" for i in range(12)],
    *[f"dqj_{i}" for i in range(12)],
    "phase",
    *[f"contact_leg_{i}" for i in range(4)],
    *[f"q_des_{i}" for i in range(12)]
]

# ======= Data Collection Loop =======
log_summary = []    


for rollout in range(MAX_ROLLOUTS):
    print(f"\n[Rollout {rollout+1}/{MAX_ROLLOUTS}] Starting...")
    robot.reset_state(q0, v0)

    
    # Sample perturbation
    delta_q = np.random.normal(0, 0.05, size=PIN_ROBOT.model.nv)
    delta_v = np.random.normal(0, 0.05, size=PIN_ROBOT.model.nv)
    delta = np.concatenate([delta_q, delta_v])
    
    # Get contact state for perturbation projection
    contacts = robot.get_current_contacts()
    contact_state = [bool(c) for c in contacts]
    
    # Compute contact Jacobian and nullspace projection
    J_c = SYSID._compute_J_c(contact_state)
    if J_c.shape[0] == 0:
        print("  No contacts - skipping perturbation")
        q_init, v_init = q0, v0
    else:
        # Construct full constraint matrix
        A_c = np.vstack([
            np.hstack([J_c, np.zeros_like(J_c)]),
            np.hstack([np.zeros_like(J_c), J_c])
        ])
        
        # Compute nullspace projection
        P = np.eye(A_c.shape[1]) - pinv(A_c) @ A_c
        delta_projected = P @ delta
        
        # Apply constrained perturbation
        delta_qc = delta_projected[:PIN_ROBOT.model.nv]
        delta_vc = delta_projected[PIN_ROBOT.model.nv:]
        q_init = pin.integrate(PIN_ROBOT.model, q0, delta_qc)
        v_init = v0 + delta_vc
        
        # Set perturbed state
        robot.reset_state(q_init, v_init)
    
    # Initialize MPC with perturbed state
    gg.q0 = q_init
    gg.x0 = np.concatenate([q_init, v_init])
    
    # Check base height
    q_check, _ = robot.get_state()
    if q_check[2] < 0.1 or q_check[2] > 0.4:
        print("  Invalid base height after projection - skipping rollout")
        log_summary.append([rollout+1, 0, "Invalid base height"])
        continue
    
    # Initialize loop variables
    sim_t = 0.0
    pln_ctr = 0
    index = 0
    success = True
    rollout_data = []
    
    print(f"  Running {TRAJ_LEN} steps ({TRAJ_LEN*SIM_DT:.1f} seconds)...")
    
    for step in range(TRAJ_LEN):
        # Get current state
        q, v = robot.get_state()
        
        # Replan at MPC frequency (20 Hz)
        if pln_ctr == 0:
            try:
                contact_config = [bool(c) for c in robot.get_current_contacts()]
                xs_plan, us_plan, f_plan = gg.optimize(
                    q, v, round(sim_t, 3), V_DES, W_DES
                )
                index = 0  # Reset plan index
            except Exception as e:
                print(f"  MPC failed at step {step}: {str(e)}")
                success = False
                log_summary.append([rollout+1, step, str(e)])
                break
        
        # Compute torque from MPC plan
        if index < len(xs_plan):
            
            # Get sensor data
            imu_gyro, imu_acc, _, _ = robot.get_imu_data()
            q_full, dq_full = robot.get_state()
            qj = q_full[7:]    # Joint positions (exclude floating base)
            dqj = dq_full[6:]  # Joint velocities (exclude floating base)
            
            # Compute phase (0-1 over gait cycle)
            phase = (sim_t % GAIT_PARAMS.gait_period) / GAIT_PARAMS.gait_period
            contact_flags = [int(c) for c in robot.get_current_contacts()]  # List of 4 ints
            
            
            tau = ctrl.id_joint_torques(
                q, v,
                xs_plan[index][:PIN_ROBOT.model.nq].copy(),
                xs_plan[index][PIN_ROBOT.model.nq:].copy(),
                us_plan[index].copy(),
                f_plan[index].copy(),
                contact_config
            )
                        
            # Apply control
            robot.send_joint_command(tau)
            
            # Compute PD setpoint (action) from torque
            q_t_full , v_t_full = robot.get_state()
            q_t = q_t_full[7:]  # Joint positions (exclude floating base)
            dq_t = v_t_full[6:]  # Joint velocities (exclude floating base)
            q_des = q_t + (tau + KV * dq_t) / KP
            
            
            # Store data point
            data_point = [
                *imu_acc, *imu_gyro, *qj, *dqj, phase, *contact_flags, *q_des
            ]
            rollout_data.append(data_point)
            
            # Failure check
            if q[2] < 0.1 or q[2] > 0.4:  # Base height out of bounds
                print(f"  Robot fell at step {step}")
                success = False
                log_summary.append([rollout+1, step, "Base height out of bounds"])
                break
        else:
            print(f"  Plan index out of bounds at step {step}")
            success = False
            log_summary.append([rollout+1, step, "Plan index out of bounds"])
            break
        
        # Update counters
        # time.sleep(SIM_DT)
        sim_t += SIM_DT
        pln_ctr = (pln_ctr + 1) % int(PLAN_FREQ / SIM_DT)
        index += 1
    
    # Save successful rollout
    if success:
        dataset.extend(rollout_data)
        log_summary.append([rollout+1, TRAJ_LEN, "Success"])
        print(f"  ✅ Collected {len(rollout_data)} samples")
    else:
        print("  ❌ Rollout failed")

# ======= Save Dataset =======
if dataset:
    print(f"\nSaving {len(dataset)} samples to CSV...")
    output_dir = "/home/anudeep/devel/workspace/data_humanoids/phase_distri"
    output_file = "go1_trot_dataset_.csv"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, output_file), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(dataset)
    print("Dataset saved successfully!")
else:
    print("No data collected - check for errors")
    
# ======= Save Log Summary =======
log_file = os.path.join(output_dir, "rollout_log_summary_.csv")
with open(log_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Rollout", "Steps", "Status"])
    writer.writerows(log_summary)
print(f"Rollout summary saved to {log_file}")