import numpy as np
import pinocchio as pin

# Load the URDF model
urdf_path = "/home/anudeep/project_r/src/unitree_ros/robots/go1_description/urdf/go1.urdf"
rmodel = pin.buildModelFromUrdf(urdf_path)
rdata = rmodel.createData()

print("Model name: " + rmodel.name)
print(rmodel)

# Define fixed joint positions (example values, adjust as needed)
q = np.array([-0.0, -0.57, -1.3, -0.0, 0.57, -1.3, 0.0, 0.57, -1.3, -0.0, 0.57, -1.3])  # replace with actual desired positions

# Set the joint velocities and accelerations to zero
v = pin.utils.zero(rmodel.nv)
a = pin.utils.zero(rmodel.nv)

# Define the gravity vector
gravity = np.array([0, 0, -9.81])  # Gravity vector

# Compute the external forces based on the gravity vector
# Assuming the mass of each joint is defined in the model, we can compute the gravitational forces
f_ext = np.zeros((rmodel.nv, 3))  # Initialize external forces array

# # Loop through each joint to calculate the gravitational force
for i in range(rmodel.nv):
    joint_mass = rmodel.inertias[i+1].mass  # Get the mass of the joint from the inertia
    f_ext[i] = joint_mass * gravity  # Calculate the gravitational force for this joint

# # Print the external forces
print("External forces: %s" % f_ext)

# Compute the joint torques using RNEA with gravity
tau = pin.rnea(rmodel, rdata, q, v, a)

# Print the joint torques
print("Joint torques for gravity compensation: %s" % tau.T)




# # Sample a random configuration
# q = pin.randomConfiguration(rmodel)
# print("q: %s" % q.T)

# v = pin.utils.zero(rmodel.nv)  # Joint velocities (set to zero)
# a = pin.utils.zero(rmodel.nv)  # Joint accelerations (set to zero)

# # Define the gravity vector
# gravity = np.array([0.0, 0.0, -9.81])  # Gravity vector in m/s^2

# # Compute the joint torques using RNEA without gravity
# tau = pin.rnea(rmodel, rdata, q, v, a)  # Call RNEA without gravity

# # If you want to include gravity, you need to set the gravity in the model
# # You can use the following line to set the gravity in the model:
# rmodel.gravity = gravity

# # Now compute the joint torques again, considering gravity
# tau_with_gravity = pin.rnea(rmodel, rdata, q, v, a)
# print("tau with gravity = ", tau_with_gravity.T)

# # Initial Robot State
# q0 = np.array(Go1Config.initial_configuration)
# q0[0:2] = 0.0  # Resetting the first two joint angles to zero
# v0 = pin.utils.zero(pin_robot.model.nv)
# x0 = np.concatenate([q0, pin.utils.zero(pin_robot.model.nv)])

# # Robot Environment Setup
# robot = PyBulletEnv(Go1Robot, q0, v0)

# while True:
#     # Get the current state of the robot
#     q, v = robot.get_state()

#     # Extract motor positions for the legs
#     fl_joint_positions = q[7:10]  # Assuming FL joints are at indices 7, 8, 9
#     fr_joint_positions = q[10:13]  # Assuming FR joints are at indices 10, 11, 12
#     rl_joint_positions = q[13:16]  # Assuming RL joints are at indices 13, 14, 15
#     rr_joint_positions = q[16:19]  # Assuming RR joints are at indices 16, 17, 18

#     # Print the FL joint positions
#     print("FL Joint Positions:", fl_joint_positions)

#     # Create a configuration vector for the FL leg
#     q_fl = np.zeros(rmodel.nq)
#     q_fl[rmodel.getJointId('FL_hip_joint')] = fl_joint_positions[0]
#     q_fl[rmodel.getJointId('FL_thigh_joint')] = fl_joint_positions[1]
#     q_fl[rmodel.getJointId('FL_calf_joint')] = fl_joint_positions[2]

#     # Compute forward kinematics for the FL leg
#     pin.forwardKinematics(rmodel, rdata, q_fl)
#     pin.updateFramePlacements(rmodel, rdata)

#     # Get the end effector position for the FL leg (FL foot)
#     fl_foot_pos = rdata.oMf[rmodel.getFrameId('FL_foot')].translation

#     # Print the end effector position
#     print("FL Foot Position:", fl_foot_pos)

#     # Optional: Add a sleep to control the loop rate
#     # time.sleep(0.01)  # Uncomment if needed to slow down the loop
