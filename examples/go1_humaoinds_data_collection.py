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

# === Configuration ===
urdf_path = Go1Config.urdf_path
robot_cfg = Go1Config()
pin_robot = robot_cfg.buildRobotWrapper()
rmodel = robot_cfg.robot_model
rdata = rmodel.createData()
sysid = SystemIdentification(
    urdf_path,
    config_file="/home/anudeep/devel/workspace/src/robot_properties_go1/src/robot_properties_go1/resources/go1_config.yaml",
    floating_base=True
)

# === Initial state ===
q0 = np.array(robot_cfg.initial_configuration)
q0[0:2] = 0.0
v0 = pin.utils.zero(rmodel.nv)
x0 = np.concatenate([q0, v0])

# === Controllers and Parameters ===
v_des = np.array([0.2, 0.04, 0.0])
w_des = 0.0
plan_freq = 0.05
sim_dt = 0.001
traj_len = 2000
max_rollouts = 10
gait_params = trot
kp = gait_params.kp
kv = gait_params.kd

# === Initialize Robot Simulation & Controller ===
gg = Go1MpcGaitGen(pin_robot, urdf_path, x0, plan_freq, q0, None)
gg.update_gait_params(gait_params, 0.0)
robot = PyBulletEnv(Go1Robot, q0, v0)
state_id = robot.saveState()
ctrl = InverseDynamicsController(pin_robot, ["FL_foot_fixed", "FR_foot_fixed", "RL_foot_fixed", "RR_foot_fixed"])
ctrl.set_gains(kp, kv)

# === Output Dataset ===
obs_to_action = []

# === Sampling Loop ===
for rollout in range(max_rollouts):
    print(f"\n[Rollout {rollout}] Starting...")

    # Restore to nominal state and apply one torque step
    robot.restoreState(state_id)
    q_tmp, v_tmp = robot.get_state()

    try:
        xs_plan, us_plan, f_plan = gg.optimize(q_tmp, v_tmp, 0.0, v_des, w_des)
    except:
        print(f"[Rollout {rollout}] NMPC failed during warm-up.")
        continue

    tau = ctrl.id_joint_torques(
        q_tmp, v_tmp,
        xs_plan[0][:rmodel.nq], xs_plan[0][rmodel.nq:],
        us_plan[0], f_plan[0], [True, True, True, True]
    )
    robot.send_joint_command(tau)
    time.sleep(0.001)

    # Get contacts after warm-up actuation
    contacts = robot.get_current_contacts()
    print(f"[Rollout {rollout}] Initial contacts: {contacts}")
    contact_schedule = [bool(c) for c in contacts]
    print(f"[Rollout {rollout}] Contact schedule: {contact_schedule}")

    A_c = sysid._compute_J_c(contact_schedule)
    if A_c.shape[0] == 0:
        print(f"[Rollout {rollout}] Skipped: no contact Jacobians available.")
        continue

    # === Sample & Project Tangent-Space Perturbations ===
    delta_q = np.random.normal(0, 0.05, size=rmodel.nv)
    delta_v = np.random.normal(0, 0.05, size=rmodel.nv)
    delta = np.concatenate([delta_q, delta_v])  # Shape = 36

    Ac_full = np.vstack([
        np.hstack([A_c, np.zeros_like(A_c)]),
        np.hstack([np.zeros_like(A_c), A_c])
    ])
    P = np.eye(Ac_full.shape[1]) - pinv(Ac_full) @ Ac_full
    delta_projected = P @ delta

    delta_qc = delta_projected[:rmodel.nv]
    delta_vc = delta_projected[rmodel.nv:]

    q_init = pin.integrate(rmodel, q0, delta_qc)
    v_init = v0 + delta_vc
    robot.reset_state(q_init, v_init)

    # === Rollout NMPC ===
    sim_t = 0.0
    pln_ctr = 0
    index = 0
    success = True

    for step in range(traj_len):
        q, v = robot.get_state()
        if pln_ctr == 0:
            contact_config = robot.get_current_contacts()
            try:
                xs_plan, us_plan, f_plan = gg.optimize(q, v, round(sim_t, 3), v_des, w_des)
            except:
                print(f"[Rollout {rollout}] NMPC failed at step {step}")
                success = False
                break

        if step < int(plan_freq / sim_dt) - 1:
            xs, us, f = xs_plan, us_plan, f_plan
        elif pln_ctr == int(plan_freq / sim_dt):
            xs, us, f = xs_plan[1:], us_plan[1:], f_plan[1:]
            index = 0
            pln_ctr = 0

        tau = ctrl.id_joint_torques(
            q, v,
            xs[index][:rmodel.nq], xs[index][rmodel.nq:],
            us[index], f[index], contact_config
        )
        robot.send_joint_command(tau)

        imu_gyro, imu_acc, _, _ = robot.get_imu_data()
        q_full, dq_full = robot.get_state()
        
        print(f"q: {q_full}")
        
        qj = q_full[7:]       # only joint positions
        dqj = dq_full[6:]     # only joint velocities
        tau_j = tau       # only joint torques

        q_des = qj + (tau_j + kv * dqj) / kp

        phase = step / traj_len
        row = imu_acc.tolist() + imu_gyro.tolist() + qj.tolist() + dqj.tolist() + [phase] + q_des.tolist()
        obs_to_action.append(row)
        
         # Log robot's forward position
        print(f"[Rollout {rollout}] Step {step}: Forward position = {q[0]}")


        if (q[2] > 0.7 or q[2] < 0.1):
            print(f"[Rollout {rollout}] Robot fell at step {step}")
            success = False
            break

        time.sleep(0.0001)
        sim_t += sim_dt
        pln_ctr += 1
        index += 1

    if success:
        print(f"[Rollout {rollout}] ✅ Collected {traj_len} samples.")
    else:
        print(f"[Rollout {rollout}] ❌ Skipped due to failure.")

# === Save to CSV ===
header = [f"imu_acc_{i}" for i in range(3)] + [f"imu_gyro_{i}" for i in range(3)] + \
         [f"qj_{i+1}" for i in range(12)] + [f"dqj_{i+1}" for i in range(12)] + ["phase"] + \
         [f"q_des_{i+1}" for i in range(12)]

save_path = "/home/anudeep/devel/workspace/data_humanoids/phase_distri/go1_actions_data_.csv"
with open(save_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(obs_to_action)

print(f"✅ Dataset saved to {save_path}")
