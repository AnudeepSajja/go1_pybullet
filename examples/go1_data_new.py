import time
import numpy as np
import pinocchio as pin
import random
from mim_data_utils import DataLogger
from robot_properties_go1.go1_wrapper import Go1Robot, Go1Config
from mpc.go1_cyclic_gen import Go1MpcGaitGen
from motions.cyclic.go1_motion import trot, jump, stand, bound, walk, trot_turn
from envs.pybullet_env import PyBulletEnv
from controllers.robot_id_controller import InverseDynamicsController
import csv
import os

# ─────────────── 1. Setup ───────────────
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

v_des = np.array([-0.44, -0.1, 0.0])  # 0.56 - jump, 0.7 - bound, 0.8 - trot
# v1 = np.array([0.38, 0.0, 0.0])
# v2 = np.array([0.56, 0.0, 0.0])

w_des = 0.0
plan_freq = 0.05
update_time = 0.0

sim_t = 0.0
sim_dt = 0.001
index = 0
pln_ctr = 0

gait_params = trot

kp = gait_params.kp
kv = gait_params.kd

# # Suggested per-joint PD gains
# kp_array = np.array([50.0, 50.0, 50.0] * 4)  # hip, thigh, knee pattern
# kd_array = np.array([4.0, 4.0, 4.0] * 4)

# Assign the arrays to kp and kv
# kp = kp_array
# kv = kd_array

lag = int(update_time/sim_dt)
gg = Go1MpcGaitGen(pin_robot, urdf_path, x0, plan_freq, q0, None)

gg.update_gait_params(gait_params, sim_t)

robot = PyBulletEnv(Go1Robot, q0, v0)

robot_id_ctrl = InverseDynamicsController(pin_robot, f_arr)
robot_id_ctrl.set_gains(gait_params.kp, gait_params.kd)

trj = 20 * 1000

simulation_time = trj + 1


foot_contacts = []
motor_positions = []
motor_velocities = []
imu_gyros = []
imu_accs = []
base_positions = []
base_velocities = []
base_orientations = []
torques_data = []
actions_data = []

foot_forces_data = []
phase_data = []


traj_length = trj


foot_contact_buffer = np.zeros((4, traj_length), dtype=np.float32)
motor_position_buffer = np.zeros((12, traj_length), dtype=np.float32)
motor_velocity_buffer = np.zeros((12, traj_length), dtype=np.float32)
imu_gyro_buffer = np.zeros((3, traj_length), dtype=np.float32)
imu_acc_buffer = np.zeros((3, traj_length), dtype=np.float32)
base_pos_buffer = np.zeros((3, traj_length), dtype=np.float32)
base_vel_buffer = np.zeros((3, traj_length), dtype=np.float32)
base_orn_buffer = np.zeros((4, traj_length), dtype=np.float32)
phase_buffer = np.zeros((1, traj_length), dtype=np.float32)

# print(motor_position_buffer.shape)
# print(foot_forces_buffer.shape)



torques_data_buffer = np.zeros((12, traj_length), dtype=np.float32)
actions_data_buffer = np.zeros((12, traj_length), dtype=np.float32)



buffer_index = 0

state_id = robot.saveState()
num_failure = 0

# robot.start_recording('go1_trot_nmpc.mp4')

for o in range(simulation_time):    

    q, v = robot.get_state()
    
    # recod the data
    
    
    imu_gyro, imu_acc, imu_pos, imu_vel = robot.get_imu_data()
    qj, dqj = q.copy(), v.copy()
    base_pos = np.array(qj[0:3])
    base_orn = np.array(qj[3:7])
    motor_pos = np.array(qj[7:])
    base_vel = np.array(dqj[0:3])
    motors_vel = np.array(dqj[6:])
    
    foot_contact = robot.get_current_contacts()
    phase = (sim_t % gait_params.gait_period) / gait_params.gait_period

    # --- Store sensor data to buffers ---
    
    motor_position_buffer[:, buffer_index] = motor_pos
    motor_velocity_buffer[:, buffer_index] = motors_vel
    imu_gyro_buffer[:, buffer_index] = imu_gyro
    imu_acc_buffer[:, buffer_index] = imu_acc
    base_pos_buffer[:, buffer_index] = base_pos
    base_vel_buffer[:, buffer_index] = base_vel
    base_orn_buffer[:, buffer_index] = base_orn
    
    foot_contact_buffer[:, buffer_index] = foot_contact
    phase_buffer[:, buffer_index] = phase
    
    # plan the torques
        
    if pln_ctr == 0:
        contact_configuration = robot.get_current_contacts()
        pr_st = time.time()
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
    
    

    # calculate the torques
    tau_t = robot_id_ctrl.id_joint_torques(q, v, xs[index][:pin_robot.model.nq].copy(), xs[index][pin_robot.model.nq:].copy() \
                                         , us[index], f[index], contact_configuration)
    # set the torques to the robot
    robot.send_joint_command(tau_t)
    
    q_t_fulll , v_t_full = robot.get_state()
    q_t = q_t_fulll[7:]
    v_t = v_t_full[6:]
    
    # compute q_des for each joints with tau_t, q_t, v_t, kp and kd
    for i in range(12):
        kpi = kp[i] if hasattr(kp, '__len__') else kp
        kvi = kv[i] if hasattr(kv, '__len__') else kv
        a_j = q_t[i] + (tau_t[i] + kvi * v_t[i]) / kpi
        actions_data_buffer[i, buffer_index] = a_j
    
    
    # Store the torques to the buffer    
    torques_data_buffer[:, buffer_index] = tau_t

    # time.sleep(0.0001)
    sim_t += sim_dt
    pln_ctr = int((pln_ctr + 1)%(plan_freq/sim_dt))
    index += 1
    
    buffer_index += 1
        
    if buffer_index==traj_length:

        buffer_index = 0
        robot.restoreState(state_id)
        
        # Collect the Data for training
        
        motor_positions.append(motor_position_buffer)
        motor_velocities.append(motor_velocity_buffer)
        imu_gyros.append(imu_gyro_buffer)
        imu_accs.append(imu_acc_buffer)
        base_positions.append(base_pos_buffer)
        base_velocities.append(base_vel_buffer)
        base_orientations.append(base_orn_buffer)
        
        foot_contacts.append(foot_contact_buffer)
        phase_data.append(phase_buffer)

        torques_data.append(torques_data_buffer)
        actions_data.append(actions_data_buffer)


        # Reset the buffer
        foot_contact_buffer = np.zeros((4, traj_length), dtype=np.float32)
        phase_buffer = np.zeros((1, traj_length), dtype=np.float32)
        
        
        motor_position_buffer = np.zeros((12, traj_length), dtype=np.float32)
        motor_velocity_buffer = np.zeros((12, traj_length), dtype=np.float32)
        imu_gyro_buffer = np.zeros((3, traj_length), dtype=np.float32)
        imu_acc_buffer = np.zeros((3, traj_length), dtype=np.float32)
        base_pos_buffer = np.zeros((3, traj_length), dtype=np.float32)
        base_vel_buffer = np.zeros((3, traj_length), dtype=np.float32)
        base_orn_buffer = np.zeros((4, traj_length), dtype=np.float32)


        torques_data_buffer = np.zeros((12, traj_length), dtype=np.float32)
        actions_data_buffer = np.zeros((12, traj_length), dtype=np.float32)


    if (q[0] > 50 or q[0] < -50 or q[1] > 50 or q[1] < -50 or q[2] > 0.7 or q[2] < 0.1):
        robot.restoreState(state_id)
        v_des = np.array([0.0, 0.0, 0.0])
        num_failure += 1
        buffer_index = 0

print("num of failure =", num_failure)

# robot.stop_recording()

# Save the data 
foot_contacts_data = np.concatenate(foot_contacts, axis=1) 
phase_data = np.concatenate(phase_data, axis=1)

motor_positions_data = np.concatenate(motor_positions, axis=1)
motor_velocities_data = np.concatenate(motor_velocities, axis=1)
imu_gyros_data = np.concatenate(imu_gyros, axis=1)
imu_accs_data = np.concatenate(imu_accs, axis=1)
base_positions_data = np.concatenate(base_positions, axis=1)
base_velocities_data = np.concatenate(base_velocities, axis=1)
base_orientations_data = np.concatenate(base_orientations, axis=1)


torques_data_buffer = np.concatenate(torques_data, axis=1)
actions_data_buffer = np.concatenate(actions_data, axis=1)

# Define the path for saving
path = "/home/anudeep/devel/workspace/src/data/trot_with_vdes/with_phase/"

# Check the shapes of all data arrays before concatenation
print("Foot Contacts Shape:", foot_contacts_data.shape)
print("Motor Positions Shape:", motor_positions_data.shape)
print("Motor Velocities Shape:", motor_velocities_data.shape)
print("IMU Gyros Shape:", imu_gyros_data.shape)
print("IMU Accs Shape:", imu_accs_data.shape)
print("Base Positions Shape:", base_positions_data.shape)
print("Base Velocities Shape:", base_velocities_data.shape)
print("Base Orientations Shape:", base_orientations_data.shape)
print("Phase variable Shape:", phase_data.shape)


print("Torques Data Shape:", torques_data_buffer.shape)
print("Actions Data Shape:", actions_data_buffer.shape)






print("kp:", kp , "kv:", kv)

# Save the data into a csv file
csv_file = path + 'go1_trot_data_actions_.csv'

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["time", "base_pos_x", "base_pos_y", "base_pos_z", "base_ori_x", "base_ori_y", "base_ori_z", "base_ori_w",
                     "base_vel_x", "base_vel_y", "base_vel_z", 
                     "imu_acc_x", "imu_acc_y", "imu_acc_z",
                     "imu_gyro_x", "imu_gyro_y", "imu_gyro_z",
                     "qj_1", "qj_2", "qj_3", "qj_4", "qj_5", "qj_6", "qj_7", "qj_8", "qj_9", "qj_10", "qj_11", "qj_12",
                     "dqj_1", "dqj_2", "dqj_3", "dqj_4", "dqj_5", "dqj_6", "dqj_7", "dqj_8", "dqj_9", "dqj_10", "dqj_11", "dqj_12",
                     "foot_1", "foot_2", "foot_3", "foot_4",
                     "phase",
                     "tau_1", "tau_2", "tau_3", "tau_4", "tau_5", "tau_6", "tau_7", "tau_8", "tau_9", "tau_10", "tau_11", "tau_12",
                     "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10", "a11", "a12",
                     "v_des_x", "v_des_y", "v_des_yaw"])
    
    for i in range(traj_length):
        writer.writerow([
            i ,
            base_positions_data[0, i], base_positions_data[1, i], base_positions_data[2, i],
            base_orientations_data[0, i], base_orientations_data[1, i], base_orientations_data[2, i], base_orientations_data[3, i],
            base_velocities_data[0, i], base_velocities_data[1, i], base_velocities_data[2, i],
            imu_accs_data[0, i], imu_accs_data[1, i], imu_accs_data[2, i],
            imu_gyros_data[0, i], imu_gyros_data[1, i], imu_gyros_data[2, i],
            *motor_positions_data[:, i],
            *motor_velocities_data[:, i],
            *foot_contacts_data[:, i],
            phase_data[0, i],
            *torques_data_buffer[:, i],
            *actions_data_buffer[:, i],
            v_des[0], v_des[1], w_des
        ])
        
# print(f"Data saved to {csv_file}")
print("Data saved successfully!")
print("Number of failures:", num_failure)   

    
