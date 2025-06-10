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
NQ = PIN_ROBOT.model.nq  # Total configuration dimension (base + joints)
NV = PIN_ROBOT.model.nv  # Total velocity dimension

# System identification setup
SYSID = SystemIdentification(
    URDF_PATH,
    config_file="/home/anudeep/devel/workspace/src/robot_properties_go1/src/robot_properties_go1/resources/go1_config.yaml",
    floating_base=True
)

# Simulation parameters
SIM_DT = 0.001
PLAN_FREQ = 0.05  # MPC replanning frequency (20 Hz)
TRAJ_LEN = 5000    # 5 seconds per rollout
MAX_ROLLOUTS = 10
V_DES = np.array([0.2, 0.04, 0.0])  # Desired velocity
W_DES = 0.0                         # Desired yaw rate

# Gait parameters
GAIT_PARAMS = trot
KP = GAIT_PARAMS.kp
KV = GAIT_PARAMS.kd
GAIT_PERIOD = GAIT_PARAMS.gait_period  # Trot gait period

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
    *[f"q_des_{i}" for i in range(12)]
]

# ======= Data Collection Loop =======
log_summary = []    

for rollout in range(MAX_ROLLOUTS):
    print(f"\n[Rollout {rollout+1}/{MAX_ROLLOUTS}] Starting...")
    robot.reset_state(q0, v0)
    
    # Sample perturbation
    delta_q = np.random.normal(0, 0.05, size=NV)
    delta_v = np.random.normal(0, 0.05, size=NV)
    delta = np.concatenate([delta_q, delta_v])
    
    # Get contact state for perturbation projection (paper uses contact constraints)
    contacts = robot.get_current_contacts()
    contact_state = [bool(c) for c in contacts]
    
    # Compute contact Jacobian and nullspace projection (Eq. 6-7 in paper)
    J_c = SYSID._compute_J_c(contact_state)
    if J_c.shape[0] == 0:
        print("  No contacts - skipping perturbation")
        q_init, v_init = q0, v0
    else:
        # Construct full constraint matrix (Eq. 7 in paper)
        A_c = np.vstack([
            np.hstack([J_c, np.zeros_like(J_c)]),
            np.hstack([np.zeros_like(J_c), J_c])
        ])
        
        # Compute nullspace projection (Eq. 6 in paper)
        P = np.eye(A_c.shape[1]) - pinv(A_c) @ A_c
        delta_projected = P @ delta
        
        # Apply constrained perturbation
        delta_qc = delta_projected[:NV]
        delta_vc = delta_projected[NV:]
        q_init = pin.integrate(PIN_ROBOT.model, q0, delta_qc)
        v_init = v0 + delta_vc
        
        # Set perturbed state
        robot.reset_state(q_init, v_init)
    
    # Initialize MPC with perturbed state
    gg.q0 = q_init
    gg.x0 = np.concatenate([q_init, v_init])
    
    # Check base height (paper filters invalid states)
    q_check, _ = robot.get_state()
    base_height = q_check[2]
    if base_height < 0.1 or base_height > 0.4:
        print(f"  Invalid base height ({base_height:.3f}m) - skipping rollout")
        log_summary.append([rollout+1, 0, f"Invalid base height: {base_height:.3f}"])
        continue
    
    # Initialize loop variables
    sim_t = 0.0
    last_mpc_time = -PLAN_FREQ  # Force MPC planning on first step
    xs_plan, us_plan, f_plan = None, None, None
    success = True
    rollout_data = []
    
    print(f"  Running {TRAJ_LEN} steps ({TRAJ_LEN*SIM_DT:.1f} seconds)...")
    
    for step in range(TRAJ_LEN):
        # Get current state
        q, v = robot.get_state()
        
        # Replan at MPC frequency (20 Hz) - paper does this
        if sim_t - last_mpc_time >= PLAN_FREQ - 1e-5:
            try:
                # Paper uses current time for gait phase alignment
                xs_plan, us_plan, f_plan = gg.optimize(
                    q, v, sim_t, V_DES, W_DES
                )
                last_mpc_time = sim_t
            except Exception as e:
                print(f"  MPC failed at step {step}: {str(e)}")
                success = False
                log_summary.append([rollout+1, step, str(e)])
                break
        
        # Compute torque from MPC plan if available
        if xs_plan is not None and len(xs_plan) > 0:
            # Time since last MPC plan
            t_elapsed = sim_t - last_mpc_time
            
            # Interpolate plan at current time (paper uses continuous trajectory)
            # MPC plans are at discrete timesteps, so we interpolate
            interp_idx = t_elapsed / PLAN_FREQ
            idx0 = min(int(np.floor(interp_idx)), len(xs_plan)-1)
            idx1 = min(idx0 + 1, len(xs_plan)-1)
            alpha = interp_idx - idx0
            
            # Linear interpolation
            x_interp = (1-alpha) * xs_plan[idx0] + alpha * xs_plan[idx1]
            u_interp = (1-alpha) * us_plan[idx0] + alpha * us_plan[idx1]
            f_interp = (1-alpha) * f_plan[idx0] + alpha * f_plan[idx1]
            
            # Paper uses contact sequence for trot: [0,1,1,0] -> [1,0,0,1]
            # Determine current contact state from gait phase
            phase_val = (sim_t % GAIT_PERIOD) / GAIT_PERIOD
            if phase_val < 0.5:
                contact_config = [False, True, True, False]  # [FL, FR, RL, RR]
            else:
                contact_config = [True, False, False, True]
            
            # Compute torque using interpolated state
            tau = ctrl.id_joint_torques(
                q, v,
                x_interp[:NQ],
                x_interp[NQ:],
                u_interp,
                f_interp,
                contact_config  # Current contact state
            )
            
            # Apply control
            robot.send_joint_command(tau)
            
            # Get sensor data - paper uses only proprioception
            imu_gyro, imu_acc, _, _ = robot.get_imu_data()
            q_full, dq_full = robot.get_state()
            
            # Joint states (7: base pos+quat, 7+12=19 joints)
            qj = q_full[7:7+12]
            dqj = dq_full[6:6+12] 
            
            # Compute PD setpoint (action) from torque - Eq.5 in paper
            q_des = qj + (tau + KV * dqj) / KP
            
            # Compute phase (0-1 over gait period) - CRITICAL CORRECTION
            phase = (sim_t % GAIT_PERIOD) / GAIT_PERIOD
            
            # Store data point
            data_point = [
                *imu_acc, *imu_gyro, *qj, *dqj, phase, *q_des
            ]
            rollout_data.append(data_point)
            
            # Failure check - paper only keeps successful rollouts
            base_height = q[2]
            if base_height < 0.1 or base_height > 0.7:
                print(f"  Robot fell at step {step} (height: {base_height:.3f}m)")
                success = False
                log_summary.append([rollout+1, step, f"Fall: {base_height:.3f}m"])
                break
        
        # Update time
        sim_t += SIM_DT
    
    # Save successful rollout - paper filters failures
    if success and rollout_data:
        dataset.extend(rollout_data)
        log_summary.append([rollout+1, TRAJ_LEN, "Success"])
        print(f"  ✅ Collected {len(rollout_data)} samples")
    else:
        print("  ❌ Rollout failed")

# ======= Save Dataset =======
if dataset:
    print(f"\nSaving {len(dataset)} samples to CSV...")
    output_dir = "/home/anudeep/devel/workspace/data_humanoids/phase_distri"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save main dataset
    output_file = "go1_trot_dataset.csv"
    with open(os.path.join(output_dir, output_file), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(dataset)
    print(f"✅ Dataset saved to {os.path.join(output_dir, output_file)}")
else:
    print("No data collected - check for errors")
    
# ======= Save Log Summary =======
log_file = os.path.join(output_dir, "rollout_log_summary.csv")
with open(log_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Rollout", "Steps", "Status"])
    writer.writerows(log_summary)
print(f"Rollout summary saved to {log_file}")