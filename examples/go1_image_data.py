import time
import numpy as np
import pinocchio as pin
import random
from mim_data_utils import DataLogger
from robot_properties_go1.go1_wrapper import Go1Robot, Go1Config
from mpc.go1_cyclic_gen import Go1MpcGaitGen
from motions.cyclic.go1_motion import trot, jump, stand, bound
from envs.pybullet_terrain_env import PyBulletTerrainEnv
from controllers.robot_id_controller import InverseDynamicsController
import csv

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

v_des = np.array([-0.7,0.2,0.0])
w_des = 0.0

plan_freq = 0.05
update_time = 0.0

sim_t = 0.0
sim_dt = 0.001
index = 0
pln_ctr = 0

gait_params = trot

lag = int(update_time/sim_dt)
gg = Go1MpcGaitGen(pin_robot, urdf_path, x0, plan_freq, q0, None)

gg.update_gait_params(gait_params, sim_t)

robot = PyBulletTerrainEnv(Go1Robot, q0, v0)

robot_id_ctrl = InverseDynamicsController(pin_robot, f_arr)
robot_id_ctrl.set_gains(gait_params.kp, gait_params.kd)

trj = 15 * 1000

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

foot_forces_data = []


traj_length = trj


foot_contact_buffer = np.zeros((4, traj_length), dtype=np.float32)
motor_position_buffer = np.zeros((12, traj_length), dtype=np.float32)
motor_velocity_buffer = np.zeros((12, traj_length), dtype=np.float32)
imu_gyro_buffer = np.zeros((3, traj_length), dtype=np.float32)
imu_acc_buffer = np.zeros((3, traj_length), dtype=np.float32)
base_pos_buffer = np.zeros((3, traj_length), dtype=np.float32)
base_vel_buffer = np.zeros((3, traj_length), dtype=np.float32)
base_orn_buffer = np.zeros((4, traj_length), dtype=np.float32)

torques_data_buffer = np.zeros((12, traj_length), dtype=np.float32)


buffer_index = 0

state_id = robot.saveState()
num_failure = 0

# Define the path for saving
path = "/home/anudeep/devel/workspace/src/data/data_with_3vel/"

# Open the CSV file once before the loop
csv_file = path + 'go1_images_with_data_.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["time", "base_pos_x", "base_pos_y", "base_pos_z", "base_ori_x", "base_ori_y", "base_ori_z", "base_ori_w",
                     "base_vel_x", "base_vel_y", "base_vel_z", 
                     "imu_acc_x", "imu_acc_y", "imu_acc_z",
                     "imu_gyro_x", "imu_gyro_y", "imu_gyro_z",
                     "qj_1", "qj_2", "qj_3", "qj_4", "qj_5", "qj_6", "qj_7", "qj_8", "qj_9", "qj_10", "qj_11", "qj_12",
                     "dqj_1", "dqj_2", "dqj_3", "dqj_4", "dqj_5", "dqj_6", "dqj_7", "dqj_8", "dqj_9", "dqj_10", "dqj_11", "dqj_12",
                     "foot_1", "foot_2", "foot_3", "foot_4",
                     "tau_1", "tau_2", "tau_3", "tau_4", "tau_5", "tau_6", "tau_7", "tau_8", "tau_9", "tau_10", "tau_11", "tau_12"])

    # Main simulation loop
    for o in range(simulation_time):  

        # Capture an image and save data for every 100th iteration  
        if o % 100 == 0:
            robot.save_image(o)

        start = time.time()
        # Get the current state of the robot
        q, v = robot.get_state()
        
        ## Plan the motion if needed
        if pln_ctr == 0:
            contact_configuration = robot.get_current_contacts()
            pr_st = time.time()
            xs_plan, us_plan, f_plan = gg.optimize(q, v, np.round(sim_t,3), v_des, w_des)

        # Use the planned motion
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

        # Save the data
        foot_contacts.append(foot_contact_buffer[:, buffer_index])
        motor_positions.append(motor_position_buffer[:, buffer_index])
        motor_velocities.append(motor_velocity_buffer[:, buffer_index])
        imu_gyros.append(imu_gyro_buffer[:, buffer_index])
        imu_accs.append(imu_acc_buffer[:, buffer_index])
        base_positions.append(base_pos_buffer[:, buffer_index])
        base_velocities.append(base_vel_buffer[:, buffer_index])
        base_orientations.append(base_orn_buffer[:, buffer_index])
        torques_data.append(torques_data_buffer[:, buffer_index])

        # Compute the joint torques using inverse dynamics
        tau = robot_id_ctrl.id_joint_torques(q, v, xs[index][:pin_robot.model.nq].copy(), xs[index][pin_robot.model.nq:].copy(), us[index], f[index], contact_configuration)
        
        # print the no of step
        print("Step number: ", o)

        # if o > 1000:
        #     v_des = np.array([0.3, 0.0, 0.0])

        # if o > 15000:
        #     v_des = np.array([0.7, 0.0, 0.0])

        # Send the joint commands to the robot
        robot.send_joint_command(tau)
        
        # Update simulation time and counters
        sim_t += sim_dt
        pln_ctr = int((pln_ctr + 1) % (plan_freq/sim_dt))
        index += 1

        # Read the robot sensors and states
        foot_contact = robot.get_current_contacts()

        
        # Get IMU data
        imu_gyro, imu_acc, imu_pos, imu_vel = robot.get_imu_data()

        # Get state data
        qj, dqj = robot.get_state()
        base_pos = np.array(qj[0:3])
        base_orn = np.array(qj[3:7])
        motor_pos = np.array(qj[7:])
        base_vel = np.array(dqj[0:3])
        motors_vel = np.array(dqj[6:])

        foot_contact_buffer[:, buffer_index] = foot_contact
        motor_position_buffer[:, buffer_index] = motor_pos
        motor_velocity_buffer[:, buffer_index] = motors_vel
        imu_gyro_buffer[:, buffer_index] = imu_gyro
        imu_acc_buffer[:, buffer_index] = imu_acc
        base_pos_buffer[:, buffer_index] = base_pos
        base_vel_buffer[:, buffer_index] = base_vel
        base_orn_buffer[:, buffer_index] = base_orn


        torques_data_buffer[:, buffer_index] = tau

        
        buffer_index += 1
            
        if buffer_index == traj_length:
            buffer_index = 0
            robot.restoreState(state_id)
            
            # Reset the buffer
            foot_contact_buffer = np.zeros((4, traj_length), dtype=np.float32)
            motor_position_buffer = np.zeros((12, traj_length), dtype=np.float32)
            motor_velocity_buffer = np.zeros((12, traj_length), dtype=np.float32)
            imu_gyro_buffer = np.zeros((3, traj_length), dtype=np.float32)
            imu_acc_buffer = np.zeros((3, traj_length), dtype=np.float32)
            base_pos_buffer = np.zeros((3, traj_length), dtype=np.float32)
            base_vel_buffer = np.zeros((3, traj_length), dtype=np.float32)
            base_orn_buffer = np.zeros((4, traj_length), dtype=np.float32)
            torques_data_buffer = np.zeros((12, traj_length), dtype=np.float32)

        # Check for failure conditions and restore state if needed
        if (q[0] > 50 or q[0] < -50 or q[1] > 50 or q[1] < -50 or q[2] > 0.7 or q[2] < 0.1):
            robot.restoreState(state_id)
            v_des = np.array([0.0, 0.0, 0.0])
            num_failure += 1

        end = time.time()
        time_for_loop = end - start

        # Write data to CSV file for each iteration
        writer.writerow([o, base_pos[0], base_pos[1], base_pos[2], base_orn[0], base_orn[1], base_orn[2], base_orn[3],
                         base_vel[0], base_vel[1], base_vel[2],
                         imu_acc[0], imu_acc[1], imu_acc[2],
                         imu_gyro[0], imu_gyro[1], imu_gyro[2],
                         motor_pos[0], motor_pos[1], motor_pos[2], motor_pos[3], motor_pos[4], motor_pos[5], motor_pos[6], motor_pos[7], motor_pos[8], motor_pos[9], motor_pos[10], motor_pos[11],
                         motors_vel[0], motors_vel[1], motors_vel[2], motors_vel[3], motors_vel[4], motors_vel[5], motors_vel[6], motors_vel[7], motors_vel[8], motors_vel[9], motors_vel[10], motors_vel[11],
                         foot_contact[0], foot_contact[1], foot_contact[2], foot_contact[3],
                         tau[0], tau[1], tau[2], tau[3], tau[4], tau[5], tau[6], tau[7], tau[8], tau[9], tau[10], tau[11]])

print("Data saved successfully!")
