## This is a demo for trot motion in mpc for Go1 Robot

import time
import numpy as np
import pinocchio as pin
import random

from mim_data_utils import DataLogger
from robot_properties_go1.go1_wrapper import Go1Robot, Go1Config
from mim_estimation.force_centroidal_ekf import ForceCentroidalEKF
from mpc.go1_cyclic_gen import Go1MpcGaitGen
from motions.cyclic.go1_motion import trot, jump, stand, bound

from envs.pybullet_env import PyBulletEnv
from controllers.robot_id_controller import InverseDynamicsController

## robot config and init
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

# trot test[0.7, 0.3, 0.0]
# jumpt_test[0.6, 0.3, 0.0]
# bound_test[0.5, 0.2, 0.0]

v_des = np.array([0.0, 0.0, 0.0])
v_des_2 = np.array([0.5, 0.3, 0.0])
w_des = 0.0
motion_name = "trot"

plan_freq = 0.05 # sec
update_time = 0.0 # sec (time of lag)

sim_t = 0.0
sim_dt = 0.001
index = 0
pln_ctr = 0

## Motion
gait_params = trot

path = "/home/khorshidi/go1_workspace/workspace/test_data/"
file_name_state = path + motion_name + "cent_state.dat"
file_name_input = path + motion_name + "input_vector.dat"

lag = int(update_time/sim_dt)
gg = Go1MpcGaitGen(pin_robot, urdf_path, x0, plan_freq, q0, None)

gg.update_gait_params(gait_params, sim_t)

robot = PyBulletEnv(Go1Robot, q0, v0)

robot_id_ctrl = InverseDynamicsController(pin_robot, f_arr)
robot_id_ctrl.set_gains(gait_params.kp, gait_params.kd)

# Create EKF instance
solo_cent_ekf = ForceCentroidalEKF(Go1Config())
R_ekf = np.diag(np.array([1e-2, 1e-6, 1e-3] + [1e1, 1e-1, 1e-1] + [1e-4, 1e-3, 1e-5]))
solo_cent_ekf.set_process_noise_cov(1e0, 1e-3, 1e-3)
solo_cent_ekf.set_measurement_noise_cov(R_ekf)

# Data Logger for collecting data
dl = DataLogger('go1_' + motion_name + '.mds')
id_contact = dl.add_field('contact_config', 4)
id_ee_force = dl.add_field('ee_force', 13)
id_cent_state = dl.add_field('cent_state', 9)
id_noisy_cent_state = dl.add_field('noisy_cent_state', 9)
id_ekf_cent_state = dl.add_field('ekf_cent_state', 9)
# dl.init_file()

simulation_time = 12000

# Arrays for data collecting for Training
cent_state = np.zeros((9, simulation_time), dtype=np.float32) 
input_vec = np.zeros((24, simulation_time), dtype=np.float32) 

cent_state_noisy = np.zeros((9, simulation_time), dtype=np.float32) 
input_vec_noisy = np.zeros((24, simulation_time), dtype=np.float32) 

# For EKF estimation
ee_contact = np.zeros((4, simulation_time), dtype=np.float32)
ee_force = np.zeros((12, simulation_time), dtype=np.float32) 
robot_q = np.zeros((19, simulation_time), dtype=np.float32)
robot_dq = np.zeros((18, simulation_time), dtype=np.float32)

state_id = robot.saveState()  # Save the initial state right after initialization
num_failure = 0
alpha = 1

for o in range(simulation_time):   
    if o == 500:
        v_des = v_des_2
    
    q, v = robot.get_state()
    if pln_ctr == 0:
        contact_configuration = robot.get_current_contacts()

        pr_st = time.time()
        xs_plan, us_plan, f_plan = gg.optimize(q, v, np.round(sim_t,3), v_des, w_des)

    # first loop assume that trajectory is planned
    if o < int(plan_freq/sim_dt) - 1:
        xs = xs_plan
        us = us_plan
        f = f_plan

    # second loop onwards lag is taken into account
    elif pln_ctr == lag and o > int(plan_freq/sim_dt)-1:
        # Not the correct logic
        # lag = int((1/sim_dt)*(pr_et - pr_st))
        lag = 0
        xs = xs_plan[lag:]
        us = us_plan[lag:]
        f = f_plan[lag:]
        index = 0

    tau = robot_id_ctrl.id_joint_torques(q, v, xs[index][:pin_robot.model.nq].copy(), xs[index][pin_robot.model.nq:].copy() \
                                         , us[index], f[index], contact_configuration)
    robot.send_joint_command(tau)

    time.sleep(0.001)
    sim_t += sim_dt
    pln_ctr = int((pln_ctr + 1)%(plan_freq/sim_dt))
    index += 1
    
    # Read the robot sensors and states
    contact_config = robot.get_current_contacts()
    ee_force_true = robot.get_ground_reaction_forces()
    ee_force_z = ee_force_true[0, 2] + ee_force_true[1, 2] + ee_force_true[2, 2] + ee_force_true[3, 2] 
    q, v = robot.get_state()
    pin.computeCentroidalMomentum(rmodel, rdata, q, v)
    actual_state = np.hstack((rdata.com[0], rdata.hg.linear, rdata.hg.angular))
        
    # EKF initial value
    if o == 0:
        solo_cent_ekf.set_mu_post(actual_state)
    
    # # Data Loging
    # dl.begin_timestep()
    # dl.log(id_contact, contact_config)
    # dl.log(id_ee_force_true, np.hstack((ee_force_true.reshape(12), ee_force_z)))
    # dl.log(id_cent_state, actual_state)
    # dl.log(id_ekf_cent_state, np.hstack((ekf_com, ekf_lin, ekf_ang)))
    # pin.computeCentroidalMomentum(rmodel, rdata, q_n, v_n)
    # measured_state = np.hstack((rdata.com[0], rdata.hg.linear, rdata.hg.angular))
    # dl.log(id_noisy_cent_state, measured_state)
    # dl.end_timestep()
    if  o >= 500:
        j = o-500
        # Collect the Data for training
        solo_cent_ekf.update_values(q, v, contact_config, ee_force_true)
        cent_state[:, j] = actual_state
        input_vec[:, j] = np.hstack((ee_force_true.reshape(12), solo_cent_ekf.get_ee_relative_positions(q)))
        
        # --------------------------------------- #
        # Read the noisy data for State Estimation
        q_n, v_n = robot.get_noisy_state()
        pin.computeCentroidalMomentum(rmodel, rdata, q_n, v_n)
        state_noisy = np.hstack((rdata.com[0], rdata.hg.linear, rdata.hg.angular))
        cent_state_noisy[:, j] = state_noisy
        
        ee_force_n = robot.get_noisy_forces()
        solo_cent_ekf.update_values(q_n, v_n, contact_config, ee_force_n)
        input_vec_noisy[:,j] = np.hstack((ee_force_n.reshape(12), solo_cent_ekf.get_ee_relative_positions(q_n)))
        
        # Save data for state estimation with EKF
        ee_contact[:, j] = contact_config
        robot_q[:, j] = q_n
        robot_dq[:,j] = v_n
        ee_force[:, j] = ee_force_n.reshape(12)
    
np.savetxt(file_name_state, cent_state, delimiter='\t')
np.savetxt(file_name_input, input_vec, delimiter='\t')

# # --------------------------------------- #
np.savetxt(path+motion_name+"cent_state_noisy.dat", cent_state_noisy, delimiter='\t')
np.savetxt(path+motion_name+"input_vector_noisy.dat", input_vec_noisy, delimiter='\t')

np.savetxt(path+motion_name+"ee_contact.dat", ee_contact, delimiter='\t')
np.savetxt(path+motion_name+"ee_force.dat", ee_force, delimiter='\t')
np.savetxt(path+motion_name+"robot_q.dat", robot_q, delimiter='\t')
np.savetxt(path+motion_name+"robot_dq.dat", robot_dq, delimiter='\t')

# dl.close_file()




