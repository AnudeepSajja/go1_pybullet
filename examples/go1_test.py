## This is a demo for trot motion in mpc for Go1 Robot

import time
import numpy as np
import pinocchio as pin
import random

from mim_data_utils import DataLogger
from robot_properties_go1.go1_wrapper import Go1Robot, Go1Config
# from mim_estimation.force_centroidal_ekf import ForceCentroidalEKF
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

v_des = np.array([0.5, 0.3, 0.0])
v_des_2 = np.array([0.7, 0.3, 0.0])
w_des = 0.0
motion_name = "bound_long_test_"

# jump velocities 
# v_des = np.array([0.2, 0.1, 0.0])
# v_des = np.array([0.4, 0.1, 0.0])
# v_des = np.array([0.3, 0.0, 0.0])j


plan_freq = 0.05 # sec
update_time = 0.0 # sec (time of lag)

sim_t = 0.0
sim_dt = 0.001
index = 0
pln_ctr = 0

## Motion
gait_params = trot

# path = "/home/khorshidi/go1_workspace/workspace/test_data/"
# file_name_state = path + motion_name + "cent_state.dat"
# file_name_input = path + motion_name + "input_vector.dat"

lag = int(update_time/sim_dt)
gg = Go1MpcGaitGen(pin_robot, urdf_path, x0, plan_freq, q0, None)

gg.update_gait_params(gait_params, sim_t)

robot = PyBulletEnv(Go1Robot, q0, v0)

robot_id_ctrl = InverseDynamicsController(pin_robot, f_arr)
robot_id_ctrl.set_gains(gait_params.kp, gait_params.kd)

# Create EKF instance
# solo_cent_ekf = ForceCentroidalEKF(Go1Config())
# R_ekf = np.diag(np.array([1e-2, 1e-6, 1e-3] + [1e1, 1e-1, 1e-1] + [1e-4, 1e-3, 1e-5]))
# solo_cent_ekf.set_process_noise_cov(1e0, 1e-3, 1e-3)
# solo_cent_ekf.set_measurement_noise_cov(R_ekf)

simulation_time = 15 * 1000 # ms

# Arrays for data collecting for Training
cent_state = [] 
input_vec = []
noisy_state = []
noisy_input = []
ee_force = []
ee_contact = []
robot_q = []
robot_dq = []

# Buffers to save reliable data only
traj_length = 2000
state_buffer = np.zeros((9, traj_length), dtype=np.float32) 
input_buffer = np.zeros((24, traj_length), dtype=np.float32)
noisy_state_buffer = np.zeros((9, traj_length), dtype=np.float32) 
noisy_input_buffer = np.zeros((24, traj_length), dtype=np.float32) 

# For EKF estimation
ee_contact_buffer = np.zeros((4, traj_length), dtype=np.float32)
ee_force_buffer = np.zeros((12, traj_length), dtype=np.float32) 
robot_q_buffer = np.zeros((19, traj_length), dtype=np.float32)
robot_dq_buffer = np.zeros((18, traj_length), dtype=np.float32) 
buffer_index = 0

state_id = robot.saveState()  # Save the initial state right after initialization
num_failure = 0
period = 2000
print("#################", motion_name)
# robot.start_recording('bound.mp4')

for o in range(simulation_time):    
    # Trot
    # if o % period == 0 : 
    #     v_des = np.array([0.1, 0.05, 0.0])
    # elif o % period == 52000:
    #     v_des = np.array([0.0, 0.0, 0.0])
    # elif o % period == 52000:
    #     v_des = np.array([0.4, 0.1, 0.0])
    # elif o % period == 52000:
    #     v_des = np.array([0.0, 0.0, 0.0])
    # elif o % period == 52000:
    #     v_des = np.array([0.3, 0.0, 0.0])
    # elif o % period == 52000:
    #     v_des = np.array([0.0, 0.0, 0.0])
    #     robot.restoreState(state_id)
    #     buffer_index = 0
    
    # Jump
    # if o % period == 0 : 
    #     v_des = np.array([0.2, 0.1, 0.0])
    # elif o % period == 12000:
    #     v_des = np.array([0.0, 0.0, 0.0])
    # elif o % period == 16000:
    #     v_des = np.array([0.4, 0.1, 0.0])
    # elif o % period == 26000:
    #     v_des = np.array([0.0, 0.0, 0.0])
    # elif o % period == 30000:
    #     v_des = np.array([0.3, 0.0, 0.0])
    # elif o % period == 40000:
    #     v_des = np.array([0.0, 0.0, 0.0])
    #     robot.restoreState(state_id)
    #     buffer_index = 0
    
    # Bound
    # if o % period == 0 : 
    #     v_des = np.array([0.3, 0.1, 0.0])
    # elif o % period == 22000:
    #     v_des = np.array([0.4, 0.1, 0.0])
    # elif o % period == 42000:
    #     v_des = np.array([0.4, 0.0, 0.0])
    # elif o % period == 62000:
    #     v_des = np.array([0.3, 0.1, 0.0])
    #     robot.restoreState(state_id)
    #     buffer_index = 0
                
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

    time.sleep(0.0001)
    sim_t += sim_dt
    pln_ctr = int((pln_ctr + 1)%(plan_freq/sim_dt))
    index += 1
    
    # EKF initial value
    if o == 2000:
        state_id = robot.saveState()
        v_des = v_des_2
    # if o > 2000:
    #     # Read the robot sensors and states
    #     contact_config = robot.get_current_contacts()
    #     ee_force_true = robot.get_ground_reaction_forces()
    #     q, v = robot.get_state()
    #     pin.computeCentroidalMomentum(rmodel, rdata, q, v)
    #     actual_state = np.hstack((rdata.com[0], rdata.hg.linear, rdata.hg.angular))
    #     solo_cent_ekf.update_values(q, v, contact_config, ee_force_true)
    #     state_buffer[:, buffer_index] = actual_state
    #     input_buffer[:, buffer_index] = np.hstack((ee_force_true.reshape(12), solo_cent_ekf.get_ee_relative_positions(q)))
    #     # --------------------------------------- #
    #     # Read the noisy data for State Estimation
    #     q_n, v_n = robot.get_noisy_state()
    #     pin.computeCentroidalMomentum(rmodel, rdata, q_n, v_n)
    #     state_noisy = np.hstack((rdata.com[0], rdata.hg.linear, rdata.hg.angular))
    #     noisy_state_buffer[:, buffer_index] = state_noisy
        
    #     ee_force_n = robot.get_noisy_forces()
    #     solo_cent_ekf.update_values(q_n, v_n, contact_config, ee_force_n)
    #     noisy_input_buffer[:, buffer_index] = np.hstack((ee_force_n.reshape(12), solo_cent_ekf.get_ee_relative_positions(q_n)))
    
    #     # Save data for state estimation with EKF
    #     ee_contact_buffer[:, buffer_index] = contact_config
    #     robot_q_buffer[:, buffer_index] = q_n
    #     robot_dq_buffer[:, buffer_index] = v_n
    #     ee_force_buffer[:, buffer_index] = ee_force_n.reshape(12)
        
    #     buffer_index += 1
        
    # if buffer_index==traj_length:
    #     buffer_index = 0
    #     # Collect the Data for training
    #     cent_state.append(state_buffer)
    #     input_vec.append(input_buffer)
    #     noisy_state.append(noisy_state_buffer)
    #     noisy_input.append(noisy_input_buffer)
    #     ee_contact.append(ee_contact_buffer)
    #     ee_force.append(ee_force_buffer)
    #     robot_q.append(robot_q_buffer)
    #     robot_dq.append(robot_dq_buffer)
        
    #     state_buffer = np.zeros((9, traj_length), dtype=np.float32) 
    #     input_buffer = np.zeros((24, traj_length), dtype=np.float32) 
    #     noisy_state_buffer = np.zeros((9, traj_length), dtype=np.float32) 
    #     noisy_input_buffer = np.zeros((24, traj_length), dtype=np.float32) 
    #     ee_contact_buffer = np.zeros((4, traj_length), dtype=np.float32)
    #     ee_force_buffer = np.zeros((12, traj_length), dtype=np.float32) 
    #     robot_q_buffer = np.zeros((19, traj_length), dtype=np.float32)
    #     robot_dq_buffer = np.zeros((18, traj_length), dtype=np.float32) 
        

    if (q[0] > 50 or q[0] < -50 or q[1] > 50 or q[1] < -50 or q[2] > 0.7 or q[2] < 0.1):
        robot.restoreState(state_id) # Restore to the last saved state
        v_des = np.array([0.0, 0.0, 0.0])
        num_failure += 1
        buffer_index = 0

# robot.stop_recording()     
print("##############    num of failure = " , num_failure, "###############")
# cent_state_np = np.hstack(cent_state)
# input_vec_np = np.hstack(input_vec)
# np.savetxt(file_name_state, cent_state_np, delimiter='\t')
# np.savetxt(file_name_input, input_vec_np, delimiter='\t')

# # # --------------------------------------- #
# noisy_state_np = np.hstack(noisy_state)
# noisy_input_np = np.hstack(noisy_input)
# ee_contact_np = np.hstack(ee_contact)
# ee_force_np = np.hstack(ee_force)
# robot_q_np = np.hstack(robot_q)
# robot_dq_np = np.hstack(robot_dq)

# np.savetxt(path+motion_name+"cent_state_noisy.dat", noisy_state_np, delimiter='\t')
# np.savetxt(path+motion_name+"input_vector_noisy.dat", noisy_input_np, delimiter='\t')

# np.savetxt(path+motion_name+"ee_contact.dat", ee_contact_np, delimiter='\t')
# np.savetxt(path+motion_name+"ee_force.dat", ee_force_np, delimiter='\t')
# np.savetxt(path+motion_name+"robot_q.dat", robot_q_np, delimiter='\t')
# np.savetxt(path+motion_name+"robot_dq.dat", robot_dq_np, delimiter='\t')




