import time
import numpy as np
import pinocchio as pin
import random
from robot_properties_go1.go1_wrapper import Go1Robot, Go1Config
from mpc.go1_cyclic_gen import Go1MpcGaitGen
from motions.cyclic.go1_motion import trot, jump, stand, bound
from envs.pybullet_env import PyBulletEnv
from controllers.robot_id_controller import InverseDynamicsController

# Initialize the robot model and URDF path
pin_robot = Go1Config.buildRobotWrapper()
urdf_path = Go1Config.urdf_path

# Create robot model and data
rmodel = Go1Config().robot_model
rdata = rmodel.createData()

# Number of effectors
n_eff = 4

# Initial configuration and velocity
q0 = np.array(Go1Config.initial_configuration)
q0[0:2] = 0.0
v0 = pin.utils.zero(pin_robot.model.nv)
x0 = np.concatenate([q0, pin.utils.zero(pin_robot.model.nv)])

# Foot contact points
f_arr = ["FL_foot_fixed", "FR_foot_fixed", "RL_foot_fixed", "RR_foot_fixed"] 

# Desired velocity and angular velocity
v_des = np.array([-0.43,-0.45,0.0])
w_des = 0.0

# Planning frequency and update time
plan_freq = 0.05
update_time = 0.0

# Simulation time and time step
sim_t = 0.0
sim_dt = 0.001
index = 0
pln_ctr = 0

# Gait parameters
gait_params = trot

# Lag for planning
lag = int(update_time/sim_dt)

# Initialize gait generator
gg = Go1MpcGaitGen(pin_robot, urdf_path, x0, plan_freq, q0, None)
gg.update_gait_params(gait_params, sim_t)

# Initialize the robot environment
robot = PyBulletEnv(Go1Robot, q0, v0)

# Initialize the inverse dynamics controller
robot_id_ctrl = InverseDynamicsController(pin_robot, f_arr)
robot_id_ctrl.set_gains(gait_params.kp, gait_params.kd)


simulation_time = 20 * 1000

# Reset flag and failure counter
reset = False
num_failure = 0

# Save the initial state of the robot
state_id = robot.saveState()

# Main simulation loop
for o in range(simulation_time):    

    start = time.time()
    # Get the current state of the robot
    q, v = robot.get_state()
    
    # Plan the motion if needed
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

    # Compute the joint torques using inverse dynamics
    tau = robot_id_ctrl.id_joint_torques(q, v, xs[index][:pin_robot.model.nq].copy(), xs[index][pin_robot.model.nq:].copy(), us[index], f[index], contact_configuration)
    
    # Send the joint commands to the robot
    robot.send_joint_command(tau)
    
    # Update simulation time and counters
    sim_t += sim_dt
    pln_ctr = int((pln_ctr + 1) % (plan_freq/sim_dt))
    index += 1

    # Save the state periodically
    if o == 500:
        state_id = robot.saveState()

    # Check for failure conditions and restore state if needed
    if (q[0] > 50 or q[0] < -50 or q[1] > 50 or q[1] < -50 or q[2] > 0.7 or q[2] < 0.1):
        robot.restoreState(state_id)
        v_des = np.array([0.0, 0.0, 0.0])
        num_failure += 1

    end = time.time()
    time_for_loop = end - start


# Print the number of failures
print("num of failure =", num_failure)
print("time for loop =", time_for_loop)
