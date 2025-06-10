# === Updated: Collect all data into a single CSV + Record Sensor BEFORE Action ===

import numpy as np
import time
import pinocchio as pin
from robot_properties_go1.go1_wrapper import Go1Robot, Go1Config
from envs.pybullet_env import PyBulletEnv
from mpc.go1_cyclic_gen import Go1MpcGaitGen
from motions.cyclic.go1_motion import trot
from controllers.robot_id_controller import InverseDynamicsController
import csv
import os
import pybullet as p

# Output path
path = "/home/anudeep/devel/workspace/src/data/khadiv_style_dataset/"
os.makedirs(path, exist_ok=True)

# Robot & Gait Setup
pin_robot = Go1Config.buildRobotWrapper()
urdf_path = Go1Config.urdf_path
robot_model = Go1Config().robot_model
robot_data = robot_model.createData()

initial_q = np.array(Go1Config.initial_configuration)
initial_q[0:2] = 0.0
initial_v = pin.utils.zero(pin_robot.model.nv)
x0 = np.concatenate([initial_q, initial_v])
f_arr = ["FL_foot_fixed", "FR_foot_fixed", "RL_foot_fixed", "RR_foot_fixed"]

# Gait and NMPC setup
gait_params = trot
plan_freq = 0.05
sim_dt = 0.001
traj_len = 15 * 1000

# Gains
kp = gait_params.kp
kd = gait_params.kd

# === Sweep over multiple v_des commands ===
v_des_list = [
    np.array([0.2, 0.0, 0.0]),
    np.array([0.5, 0.0, 0.0]),
    np.array([0.8, 0.0, 0.0]),
    np.array([0.6, 0.2, 0.0]),
    np.array([0.4, 0.0, 0.5])
]
n_trajectories_per_vdes = 5

# Unified log
master_log = []

for v_des in v_des_list:
    for traj_idx in range(n_trajectories_per_vdes):
        # === Perturb Initial State ===
        q0 = initial_q.copy()
        q0[0:2] += np.random.uniform(-0.1, 0.1, size=2)  # perturb x, y base position
        v0 = initial_v.copy()
        q0[2] += np.random.uniform(-0.02, 0.02)
        q0[3:7] += np.random.uniform(-0.01, 0.01, size=4)
        q0[7:] += np.random.uniform(-0.1, 0.1, size=12)
        v0[6:] += np.random.uniform(-0.5, 0.5, size=12)

        robot = PyBulletEnv(Go1Robot, q0, v0)
        robot_id_ctrl = InverseDynamicsController(pin_robot, f_arr)
        robot_id_ctrl.set_gains(kp, kd)
        gait_gen = Go1MpcGaitGen(pin_robot, urdf_path, x0, plan_freq, q0, None)
        gait_gen.update_gait_params(gait_params, 0.0)

        index = 0
        sim_t = 0.0
        fail = False

        for step in range(traj_len):
            # === Record state BEFORE applying action ===
            q, v = robot.get_state()
            imu_gyro, imu_acc, _, _ = robot.get_imu_data()
            contact = robot.get_current_contacts()

            # NMPC planning (if needed)
            if step % int(plan_freq / sim_dt) == 0:
                xs, us, fs = gait_gen.optimize(q, v, round(sim_t, 3), v_des, 0.0)
                index = 0

            # Compute torque from expert policy
            tau = robot_id_ctrl.id_joint_torques(
                q, v,
                xs[index][:pin_robot.model.nq].copy(),
                xs[index][pin_robot.model.nq:].copy(),
                us[index], fs[index], contact
            )

            motor_pos = q[7:]
            motor_vel = v[6:]
            a = motor_pos + (tau + kd * motor_vel) / kp

            # Log row
            master_log.append(np.concatenate([
                imu_acc, imu_gyro, motor_pos, motor_vel, contact, tau, a, v_des
            ]))

            robot.send_joint_command(tau)

            if q[2] < 0.1 or q[2] > 0.7:
                print(f"[Warning] Trajectory {traj_idx} with v_des {v_des} failed at step {step}.")
                fail = True
                break

            index += 1
            sim_t += sim_dt
            time.sleep(sim_dt)

        p.disconnect()

# Save unified log
header = [
    "imu_acc_x", "imu_acc_y", "imu_acc_z", "imu_gyro_x", "imu_gyro_y", "imu_gyro_z",
    *[f"qj_{i+1}" for i in range(12)], *[f"dqj_{i+1}" for i in range(12)],
    "foot_1", "foot_2", "foot_3", "foot_4",
    *[f"tau_{i+1}" for i in range(12)], *[f"a{i+1}" for i in range(12)],
    "v_des_x", "v_des_y", "v_des_yaw"
]

out_path = os.path.join(path, "go1_nmpc_sensor_dataset.csv")
np.savetxt(out_path, np.array(master_log), delimiter=",", header=",".join(header), comments='')
print(f"[INFO] All data saved to {out_path}")
