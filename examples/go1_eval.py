import time
import numpy as np
import pinocchio as pin
from robot_properties_go1.go1_wrapper import Go1Robot, Go1Config
from mpc.go1_cyclic_gen import Go1MpcGaitGen
from motions.cyclic.go1_motion import trot
from envs.pybullet_env import PyBulletEnv
from controllers.robot_id_controller import InverseDynamicsController
import torch
from NMPCPredictor import NMPCPredictor 

import matplotlib.pyplot as plt



# Robot Configuration and Initialization
pin_robot = Go1Config.buildRobotWrapper()
urdf_path = Go1Config.urdf_path

# Initial Robot State
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
min = 0  # minutes
sec = 15  # seconds

simulation_time = (h * 60 * 60 * 1000) + (min * 60 * 1000) + (sec * 1000)  # milliseconds
state_id = robot.saveState()
num_failure = 0

# Lists to store predicted actions and actual torques for plotting later
predicted_actions_list = []
actual_torques_list = []

end2end_test_model = '/home/anudeep/devel/workspace/models/end2end_test_entire.pth'

estimator = torch.load(end2end_test_model, map_location=torch.device('cpu'))

try:
    # Simulation Loop
    for o in range(simulation_time):
        if o == 500:
            state_id = robot.saveState()

        if o > 500:

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

            # IMU Data and state data retrieval
            imu_gyro, imu_acc, imu_pos, imu_vel = robot.get_imu_data()
            qj, dqj = robot.get_state()
            base_pos = np.array(qj[0:3])
            base_orn = np.array(qj[3:7])
            base_vel = np.array(dqj[0:3])
            motor_pos = np.array(qj[7:])
            motors_vel = np.array(dqj[6:])
            foot_contact = robot.get_current_contacts()
            
            estimator_input = np.concatenate([np.array([base_pos[2]]), base_orn, base_vel, imu_acc, imu_gyro, motor_pos, motors_vel, foot_contact])
            estimator_inputtensor = torch.tensor(estimator_input, dtype=torch.float32)

            # Get the predicted state from the estimator model
            estimator.eval()
            with torch.no_grad():
                predicted_torques = estimator(estimator_inputtensor)

            predicted_torques_numpy = predicted_torques.cpu().numpy()  

            tau = robot_id_ctrl.id_joint_torques(q, v,
                                                xs[pln_ctr][:pin_robot.model.nq].copy(),
                                                xs[pln_ctr][pin_robot.model.nq:].copy(), 
                                                us[pln_ctr], f[pln_ctr],
                                                contact_configuration)

            # Store the values for plotting later
            predicted_actions_list.append(predicted_torques_numpy)
            actual_torques_list.append(tau)

            # Send joint commands to the robot (you can choose to send either predicted or actual torques)
            # robot.send_joint_command(tau)
            robot.send_joint_command(predicted_torques_numpy)

            end = time.time()
            time_to_sleep = max(0.0001 - (end - start), 0)   # collect data at ~10kHz frequency (10ms)
            time.sleep(time_to_sleep)   # Sleep to maintain the simulation frequency

            sim_t += sim_dt
            pln_ctr = int((pln_ctr + 1) % (plan_freq / sim_dt))

except KeyboardInterrupt:
    print("Simulation interrupted by user.")

# Create a directory to save plots if it doesn't exist.
save_directory = '/home/anudeep/devel/workspace/src/data/plots'   # Specify your desired directory here

# Create subplots for each joint.
predicted_actions_array = np.array(predicted_actions_list)
actual_torques_array = np.array(actual_torques_list)

num_joints = predicted_actions_array.shape[1] if len(predicted_actions_array.shape) > 1 else len(predicted_actions_array)
figures_list=[]

for i in range(num_joints):
    plt.figure(figsize=(8,6))
    plt.plot(predicted_actions_array[:, i], label='Predicted Torques', color='blue')
    plt.plot(actual_torques_array[:, i], label='Actual Torque', color='orange', linestyle='--')
    
    plt.title(f'Joint {i+1} Torque Comparison')
    plt.xlabel('Time Step')
    plt.ylabel('Torque')
    plt.legend()
    
    # Save each figure in the specified directory.
    plt.savefig(f'{save_directory}joint_{i+1}_torque_comparison.png')
    figures_list.append(f'joint_{i+1}_torque_comparison.png') 

plt.show() 

print("Figures saved:", figures_list)
print("##############    Number of failures =", num_failure, "###############")
