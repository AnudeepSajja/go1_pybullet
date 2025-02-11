import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data
import time


# Connect to PyBullet in GUI mode
p.connect(p.GUI)

# Set the search path for URDF files
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load the ground plane, robot (R2D2), and a larger cube
p.loadURDF('plane.urdf')
# robot_id = p.loadURDF("r2d2.urdf", [0, 0, 0.4])
urdf_path = "/home/anudeep/devel/workspace/src/robot_properties_go1/src/robot_properties_go1/resources/urdf/go1_minimal.urdf"
robot_id = p.loadURDF(urdf_path, [0, 0, 0.4])
# cube_id = p.loadURDF('cube.urdf', basePosition=[2.0, 0, 0.4], globalScaling=1)  # Increase size of the cube

# Set gravity
p.setGravity(0, 0, -9.81)

num_links = p.getNumJoints(robot_id)

camera_link_name = "camera_joint_face"
camera_link_index = None

for i in range(num_links):
    joint_info = p.getJointInfo(robot_id, i)
    if joint_info[1].decode("utf-8") == camera_link_name:
        camera_link_index = joint_info[0]  # Use joint index if it's a joint
        break

if camera_link_index is None:
    print(f"Camera link '{camera_link_name}' not found!")



while True:
    p.stepSimulation()
    # time.sleep(1./240.)

    link_state = p.getLinkState(robot_id, camera_link_index)
    cameraEye = link_state[0]  # Position
    forwardDir = [1, 0, 0]  # Forward direction

    cameraTarget = [cameraEye[0] + forwardDir[0],
                    cameraEye[1] + forwardDir[1],
                    cameraEye[2] + forwardDir[2]]

    cameraUp = [0, 0, 1]  # Up direction

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



