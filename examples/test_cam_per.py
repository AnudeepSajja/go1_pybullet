import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data
import time
import math

import transforms3d

# Connect to PyBullet in GUI mode
p.connect(p.GUI)

# Set the search path for URDF files
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load the ground plane and your robot URDF
p.loadURDF('plane.urdf')
urdf_path = "/home/anudeep/devel/workspace/src/robot_properties_go1/src/robot_properties_go1/resources/urdf/go1_minimal.urdf"
robot_id = p.loadURDF(urdf_path, [0, 0, 0.4])

cube_id = p.loadURDF('cube.urdf', basePosition=[2.5, 0, 0.4], globalScaling=1)  # Increase size of the cube


# Set gravity
p.setGravity(0, 0, -9.81)

# Print available links to debug
num_links = p.getNumJoints(robot_id)
print("Available links:")
for i in range(num_links):
    joint_info = p.getJointInfo(robot_id, i)
    print(joint_info[1].decode("utf-8"))  # Print link names

# Get the index of the camera link (assuming it's named 'CameraTop_optical_frame')
camera_link_name = "camera_joint_face"
camera_link_index = None

for i in range(num_links):
    joint_info = p.getJointInfo(robot_id, i)
    if joint_info[1].decode("utf-8") == camera_link_name:
        camera_link_index = joint_info[0]  # Use joint index if it's a joint
        break

if camera_link_index is None:
    print(f"Camera link '{camera_link_name}' not found!")

# Get camera view from the camera link
link_state = p.getLinkState(robot_id, camera_link_index)
# print(link_state)

# Unpack the returned values
cameraEye = link_state[0]  # Position
# camera_rotations = p.getEulerFromQuaternion(link_state[1]) 

# print(f"Camera position: {cameraEye}")
# print(f"Camera orientation: {camera_rotations}")

# x = math.cos(camera_rotations[1]) 
# y = math.sin(camera_rotations[1]) 

# camera_target = [x, y, 0.0]
# camera_target = camera_orn

forwardDirection = [0.1, 0.0, 0.0]  # Change this vector based on your desired direction

# Calculate the new camera target
cameraTarget = [cameraEye[0] + forwardDirection[0],
                cameraEye[1] + forwardDirection[1],
                cameraEye[2] + forwardDirection[2]]

# Define up direction
cameraUp = [0.0, 0.0, 1.0]

width = 640
height = 480
fov = 120
aspect = width / height
near = 0.02
far = 15

view_matrix = p.computeViewMatrix(cameraEye, cameraTarget, cameraUp)
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

images = p.getCameraImage(width,
                            height,
                            view_matrix,
                            projection_matrix,
                            shadow=True,
                            renderer=p.ER_BULLET_HARDWARE_OPENGL)

rgb_opengl = np.reshape(images[2], (height, width, 4)) * 1. / 255.

# # Display the RGB image using Matplotlib
# plt.imshow(rgb_opengl)
# plt.title('RGB Image from OpenGL Renderer')
# plt.axis('off')  # Hide axes for better visualization
# plt.show()

while True:
    print("Running")
