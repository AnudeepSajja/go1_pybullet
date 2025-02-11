import time
import numpy as np
import pinocchio as pin
import random
from mim_data_utils import DataLogger
from robot_properties_go1.go1_wrapper import Go1Robot, Go1Config
from mpc.go1_cyclic_gen import Go1MpcGaitGen
from motions.cyclic.go1_motion import trot, jump, stand, bound
from envs.pybullet_env import PyBulletEnv
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

v_des = np.array([0.5, 0.0, 0.0])
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

robot = PyBulletEnv(Go1Robot, q0, v0)

robot_id_ctrl = InverseDynamicsController(pin_robot, f_arr)
robot_id_ctrl.set_gains(gait_params.kp, gait_params.kd)

simulation_time = 15 * 1000


foot_contacts = []
motor_positions = []
motor_velocities = []
imu_gyros = []
imu_accs = []
base_positions = []
base_velocities = []
base_orientations = []
torques_data = []

traj_length = 2000
reset = False
buffer = 500

foot_contacts = []
motor_positions = []
motor_velocities = []
imu_gyros = []
imu_accs = []
base_positions = []
base_velocities = []
base_orientations = []
torques_data = []


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
num_trajectory = 0

for o in range(simulation_time):    
    q, v = robot.get_state()
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

    tau = robot_id_ctrl.id_joint_torques(q, v, xs[index][:pin_robot.model.nq].copy(), xs[index][pin_robot.model.nq:].copy() \
                                         , us[index], f[index], contact_configuration)
    robot.send_joint_command(tau)

    time.sleep(0.0001)
    sim_t += sim_dt
    pln_ctr = int((pln_ctr + 1)%(plan_freq/sim_dt))
    index += 1

    if o == 500:
        state_id = robot.saveState()

    if o > 500:
        buffer_index += 1
        
        
    if buffer_index==traj_length:
        buffer_index = 0
        robot.restoreState(state_id)
        reset = True
        num_trajectory += 1

    if reset == True and buffer!=0:
        v_des = np.array([0.0, 0.0, 0.0])
        buffer -= 1
    else:
        v_des = np.array([0.5, 0.0, 0.0])
        reset = False
        buffer = 500

print("Total number of trajectories: ", num_trajectory)

        





