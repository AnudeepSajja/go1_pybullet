## This is a demo for trot motion in mpc for Go1 Robot

import time
import numpy as np
import pinocchio as pin

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

v_des = np.array([0.3, 0.2, 0.0])
v_des_2 = np.array([0.5, 0.2, 0.0])
w_des = 0.0

path = "/home/khorshidi/go1_workspace/workspace/"

plan_freq = 0.05 # sec
update_time = 0.0 # sec (time of lag)

sim_t = 0.0
sim_dt = .001
index = 0
pln_ctr = 0

## Motion
gait_params = bound
motion_name = "jump"
lag = int(update_time/sim_dt)
gg = Go1MpcGaitGen(pin_robot, urdf_path, x0, plan_freq, q0, None)

gg.update_gait_params(gait_params, sim_t)

robot = PyBulletEnv(Go1Robot, q0, v0)

robot_id_ctrl = InverseDynamicsController(pin_robot, f_arr)
robot_id_ctrl.set_gains(gait_params.kp, gait_params.kd)

# Create EKF instance
solo_cent_ekf = ForceCentroidalEKF(Go1Config())

simulation_time = 20 * 1000 # ms

# Arrays for data collecting for Training
actual_cent_state = np.zeros((9, simulation_time), dtype=np.float32) 
measured_cent_state = np.zeros((9, simulation_time), dtype=np.float32) 

ee_force = np.zeros((12, simulation_time), dtype=np.float32) 
ee_r = np.zeros((12, simulation_time), dtype=np.float32)
contact_config = np.zeros((4, simulation_time))

for o in range(simulation_time):
    if o > 10000: 
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

    # time.sleep(0.001)
    sim_t += sim_dt
    pln_ctr = int((pln_ctr + 1)%(plan_freq/sim_dt))
    index += 1
    
    # Read the robot sensors and states
    contact_schedule = robot.get_current_contacts()
    q, v = robot.get_state()
    pin.computeCentroidalMomentum(rmodel, rdata, q, v)
    actual_state = np.hstack((rdata.com[0], rdata.hg.linear, rdata.hg.angular))
    actual_cent_state[:, o] = actual_state
    # EKF initial value
    if o == 0:
        solo_cent_ekf.set_mu_post(actual_state)
    
    # Collect the Data for training
    
    # --------------------------------------- #
    # Read the noisy data for State Estimation
    q_n, v_n = robot.get_noisy_state()
    pin.computeCentroidalMomentum(rmodel, rdata, q_n, v_n)
    state_noisy = np.hstack((rdata.com[0], rdata.hg.linear, rdata.hg.angular))
    measured_cent_state[:, o] = state_noisy
    
    ee_force_n = robot.get_noisy_forces()
    solo_cent_ekf.update_values(q_n, v_n, contact_schedule, ee_force_n)
    ee_force[:, o] = ee_force_n.reshape(12)
    
    ee_r[:, o] = solo_cent_ekf.get_ee_relative_positions(q_n)
    # Save data for state estimation with EKF
    contact_config[:, o] = contact_schedule

np.savetxt(path+"actual_cent_state", actual_cent_state, delimiter='\t')
np.savetxt(path+"measured_cent_state.dat", measured_cent_state, delimiter='\t')

np.savetxt(path+"ee_force.dat", ee_force, delimiter='\t')
np.savetxt(path+"ee_r.dat", ee_r, delimiter='\t')
np.savetxt(path+"contact_config.dat", contact_config, delimiter='\t')


# dl.close_file()




