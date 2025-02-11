import time
import numpy as np
import pinocchio as pin
import csv

from robot_properties_go1.go1_wrapper import Go1Robot, Go1Config
from mpc.go1_cyclic_gen import Go1MpcGaitGen
from motions.cyclic.go1_motion import trot
from envs.pybullet_env import PyBulletEnv
from controllers.robot_id_controller import InverseDynamicsController

# Robot Configuration and Initialization
pin_robot = Go1Config.buildRobotWrapper()
urdf_path = Go1Config.urdf_path
rmodel = Go1Config().robot_model
rdata = rmodel.createData()

# Initial Robot State
n_eff = 4
q0 = np.array(Go1Config.initial_configuration)
q0[0:2] = 0.0  # Resetting the first two joint angles to zero
v0 = pin.utils.zero(pin_robot.model.nv)
x0 = np.concatenate([q0, pin.utils.zero(pin_robot.model.nv)])

# Foot Fixed Names
f_arr = ["FL_foot_fixed", "FR_foot_fixed", "RL_foot_fixed", "RR_foot_fixed"]

# Desired Velocities
v_des = np.array([0.5, 0.3, 0.0])
w_des = 0.0
motion_name = "trot"

# Planning Parameters
plan_freq = 0.05  # seconds 
update_time = 0.0  # seconds (time of lag)
lag = int(update_time / 0.001)  # Calculate lag based on simulation time step

# Simulation Parameters
sim_t = 0.0
sim_dt = 0.001
pln_ctr = 0

# Motion Generation
gait_params = trot
gg = Go1MpcGaitGen(pin_robot, urdf_path, x0, plan_freq, q0, None)
gg.update_gait_params(gait_params, sim_t)

# Robot Environment Setup
robot = PyBulletEnv(Go1Robot, q0, v0)
robot_id_ctrl = InverseDynamicsController(pin_robot, f_arr)
robot_id_ctrl.set_gains(gait_params.kp, gait_params.kd)

# Simulation Time
h = 0  # hours
min = 30  # minutes
sec = 0  # seconds

simulation_time = (h * 60 * 60 * 1000) + (min * 60 * 1000) + (sec * 1000)  # milliseconds
state_id = robot.saveState()
num_failure = 0

# Buffer Time Setup
buffer_time = 30000 # 30 seconds
buffer_start_time = 0  # Initialize the buffer start time
data_collected = False  # Flag to track if data is being collected

# Unstable States (if needed)
unstable_states = [
    np.array([1, 0, 0, 0]),
    np.array([0, 1, 0, 0]),
    np.array([0, 0, 1, 0]),
    np.array([0, 0, 0, 1]),
    np.array([0, 0, 0, 0]),
]

# CSV File Setup
csv_file = '/home/anudeep/devel/workspace/src/data/go1_trot.csv'
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

    # Simulation Loop
    for o in range(simulation_time):
        if o == 750:
            state_id = robot.saveState()

        start = time.time()
        q, v = robot.get_state()

        # Plan trajectory
        if pln_ctr == 0:
            contact_configuration = robot.get_current_contacts()
            xs_plan, us_plan, f_plan = gg.optimize(q, v, np.round(sim_t, 3), v_des, w_des)

        # Select trajectory based on planning counter
        if o < int(plan_freq / sim_dt) - 1:
            xs, us, f = xs_plan, us_plan, f_plan
        elif pln_ctr == lag and o > int(plan_freq / sim_dt) - 1:
            lag = 0
            xs = xs_plan[lag:]
            us = us_plan[lag:]
            f = f_plan[lag:]

        # Calculate joint torques
        tau = robot_id_ctrl.id_joint_torques(q, v, xs[pln_ctr][:pin_robot.model.nq].copy(),
                                              xs[pln_ctr][pin_robot.model.nq:].copy(), us[pln_ctr], f[pln_ctr],
                                              contact_configuration)

        # Get IMU data
        imu_gyro, imu_acc, imu_pos, imu_vel = robot.get_imu_data()

        # Get state data
        qj, dqj = robot.get_state()
        base_pos = np.array(qj[0:3])
        base_orn = np.array(qj[3:7])
        motor_pos = np.array(qj[7:])
        base_vel = np.array(dqj[0:3])
        motors_vel = np.array(dqj[6:])
        foot_contact = robot.get_current_contacts()

        # Send joint commands
        robot.send_joint_command(tau)

        # Check if we are within the buffer time to collect data
        if buffer_start_time < buffer_time:
            # Write data to CSV
            writer.writerow([sim_t, base_pos[0], base_pos[1], base_pos[2], base_orn[0], base_orn[1], base_orn[2], base_orn[3],
                             base_vel[0], base_vel[1], base_vel[2],
                             imu_acc[0], imu_acc[1], imu_acc[2],
                             imu_gyro[0], imu_gyro[1], imu_gyro[2],
                             *motor_pos,  # Unpack motor positions
                             *motors_vel,  # Unpack motors velocities
                             *foot_contact,  # Unpack floor contacts
                             *tau])  # Unpack torques
            buffer_start_time += sim_dt * 1000  # Increment the buffer time in milliseconds
        else:
            # Buffer time is reached, reset the robot and stop data collection
            robot.restoreState(state_id)
            # num_failure += 1
            buffer_start_time = 0  # Reset the buffer start time

        # time.sleep(0.0001)
        end = time.time()
        time_to_sleep = 0.0001 - (end - start)  # collect data at 1kHz
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)  # Sleep to maintain the simulation frequency

        sim_t += sim_dt
        pln_ctr = int((pln_ctr + 1) % (plan_freq / sim_dt))

        # Check for out-of-bounds conditions
        if (q[0] > 20 or q[0] < -20 or q[1] > 20 or q[1] < -20 or q[2] > 0.7 or q[2] < 0.1):
            robot.restoreState(state_id)
            num_failure += 1

print("##############    Number of failures = ", num_failure, "###############")
