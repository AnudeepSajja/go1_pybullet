## This is a demo for trot motion in mpc
## Author : Avadesh Meduri & Paarth Shah
## Date : 21/04/2021

import time
import numpy as np
import pinocchio as pin

from mim_data_utils import DataLogger
from robot_properties_solo.solo12wrapper import Solo12Robot, Solo12Config
from mim_estimation.force_centroidal_ekf import ForceCentroidalEKF
from mpc.abstract_cyclic_gen import SoloMpcGaitGen
from motions.cyclic.solo12_trot import trot, trot_turn
from motions.cyclic.solo12_jump import jump
from motions.cyclic.solo12_bound import bound

from envs.pybullet_env import PyBulletEnv
from controllers.robot_id_controller import InverseDynamicsController

## robot config and init
pin_robot = Solo12Config.buildRobotWrapper()
urdf_path = Solo12Config.urdf_path

rmodel = Solo12Config().robot_model
rdata = rmodel.createData()

n_eff = 4
q0 = np.array(Solo12Config.initial_configuration)
q0[0:2] = 0.0

v0 = pin.utils.zero(pin_robot.model.nv)
x0 = np.concatenate([q0, pin.utils.zero(pin_robot.model.nv)])
f_arr = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]

v_des = np.array([0.5,0.0,0.0])
w_des = 0.0

plan_freq = 0.05 # sec
update_time = 0.0 # sec (time of lag)

sim_t = 0.0
sim_dt = .001
index = 0
pln_ctr = 0

## Motion
gait_params = trot
motion_name = "trot"
lag = int(update_time/sim_dt)
gg = SoloMpcGaitGen(pin_robot, urdf_path, x0, plan_freq, q0, None)

gg.update_gait_params(gait_params, sim_t)

robot = PyBulletEnv(Solo12Robot, q0, v0)
robot_id_ctrl = InverseDynamicsController(pin_robot, f_arr)
robot_id_ctrl.set_gains(gait_params.kp, gait_params.kd)

plot_time = 0 #Time to start plotting

solve_times = []

# Create EKF instance
solo_cent_ekf = ForceCentroidalEKF(Solo12Config())
solo_cent_ekf.set_process_noise_cov(1e0, 1e-3, 1e-3)
R_ekf = np.diag(np.array([1e-2, 1e-6, 1e-9] + [1e-4, 1e-4, 1e-2] + [1e-6, 1e-6, 1e-3]))
solo_cent_ekf.set_measurement_noise_cov(R_ekf)

# Data Logger for collecting data
dl = DataLogger('solo12_' + motion_name + '.mds')
id_contact = dl.add_field('contact_config', 4)
id_ee_force = dl.add_field('ee_force', 13)
id_cent_state = dl.add_field('cent_state', 9)
id_noisy_cent_state = dl.add_field('noisy_cent_state', 9)
id_ekf_cent_state = dl.add_field('ekf_cent_state', 9)
dl.init_file()

simulation_time = 10 * 1000
# Arrays for data collecting for Training
cent_state = np.zeros((9, simulation_time)) 
input_vec = np.zeros((24, simulation_time)) 


for o in range(simulation_time):
    # this bit has to be put in shared memory
    q, v = robot.get_state()
    
    # if o == int(50*(plan_freq/sim_dt)):
    #     gait_params = trot
    #     gg.update_gait_params(gait_params, sim_t)
    #     robot_id_ctrl.set_gains(gait_params.kp, gait_params.kd)

    if pln_ctr == 0:
        contact_configuration = robot.get_current_contacts()
        
        pr_st = time.time()
        xs_plan, us_plan, f_plan = gg.optimize(q, v, np.round(sim_t,3), v_des, w_des)

        # Plot if necessary
        # if sim_t >= plot_time:
            # gg.plot_plan(q, v)
            # gg.save_plan("trot")

        pr_et = time.time()
        solve_times.append(pr_et - pr_et)

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

    tau = robot_id_ctrl.id_joint_torques(q, v, xs[index][:pin_robot.model.nq].copy(), xs[index][pin_robot.model.nq:].copy()\
                                , us[index], f[index], contact_configuration)
    robot.send_joint_command(tau)

    # time.sleep(0.001)
    sim_t += sim_dt
    pln_ctr = int((pln_ctr + 1)%(plan_freq/sim_dt))
    index += 1

    # State Estimation
    contact_config = robot.get_current_contacts()
    ee_force = robot.get_ground_reaction_forces()
    ee_force_z = ee_force[0, 2] + ee_force[1, 2] + ee_force[2, 2] + ee_force[3, 2] 
    q, v = robot.get_state()
    pin.computeCentroidalMomentum(rmodel, rdata, q, v)
    
    # EKF initial value
    if o == 0:
        solo_cent_ekf.set_mu_post(np.hstack((rdata.com[0], rdata.hg.linear, rdata.hg.angular)))
        
    # EKF update
    q_n, v_n = robot.get_noisy_state()
    ee_force_n = robot.get_noisy_forces()
    # solo_cent_ekf.update_filter(q_n, v_n, contact_config, ee_force_n, integration_method="rk4")
    ekf_com, ekf_lin, ekf_ang = solo_cent_ekf.get_filter_output()
    # ee_pos = solo_cent_ekf.get_ee_relative_positions()
    
    # Data Loging with Logger
    dl.begin_timestep()
    dl.log(id_contact, contact_config)
    dl.log(id_ee_force, np.hstack((ee_force.reshape(12), ee_force_z)))
    dl.log(id_cent_state, np.hstack((rdata.com[0], rdata.hg.linear, rdata.hg.angular)))
    dl.log(id_ekf_cent_state, np.hstack((ekf_com, ekf_lin, ekf_ang)))
    pin.computeCentroidalMomentum(rmodel, rdata, q_n, v_n)
    dl.log(id_noisy_cent_state, np.hstack((rdata.com[0], rdata.hg.linear, rdata.hg.angular)))
    dl.end_timestep()
    
    # Collect the Data for training
    # cent_state[:, o] = np.hstack((rdata.com[0], rdata.hg.linear, rdata.hg.angular))
    # input_vec[:, o] = np.hstack((ee_force.reshape(12), ee_pos))    
# np.savetxt("cent_state.dat", cent_state, delimiter='\t')
# np.savetxt("input_vec.dat", input_vec, delimiter='\t')
dl.close_file()



