
import pybullet
from bullet_utils.env import BulletEnvWithGround
import random
import numpy as np
import os
from PIL import Image

class PyBulletTerrainEnv:

    def __init__(self, robot, q0, v0):

        print("loading bullet")
        self.env = BulletEnvWithGround()

        # self.height_field_terrain()

        self.robot = self.env.add_robot(robot())
        self.robot.reset_state(q0, v0)

        self.camera_link_name = "camera_joint_face"
        self.camera_link_index = None
        self.setup_camera()
        
        # pybullet.resetDebugVisualizerCamera( cameraDistance=-1.5, cameraYaw=45, cameraPitch=-10, cameraTargetPosition=[0,0,.8])
        ## For data recording
        self.q_arr = []
        self.v_arr = []

    def height_field_terrain(self):
        numHeightfieldRows = 50
        numHeightfieldColumns = 100
        heightPerturbationRange = 0.035

        position = [1.7,0,0]
        
        # Generate heightfield data with random perturbations
        heightfieldData = [random.uniform(0, heightPerturbationRange) for _ in range(numHeightfieldRows * numHeightfieldColumns)]
        
        # Create the terrain shape
        terrainShape = pybullet.createCollisionShape(
            shapeType=pybullet.GEOM_HEIGHTFIELD,
            meshScale=[0.04, 0.04, 0.5],
            heightfieldTextureScaling=(numHeightfieldRows - 1) / 2,
            heightfieldData=heightfieldData,
            numHeightfieldRows=numHeightfieldRows,
            numHeightfieldColumns=numHeightfieldColumns
        )
        
        # Create the terrain body
        terrain = pybullet.createMultiBody(
            baseMass=0.1, 
            baseCollisionShapeIndex=terrainShape,
            basePosition=position, 
            baseOrientation=pybullet.getQuaternionFromEuler([0, 0, 0])
        )
    
    def add_robot(self, robot, q0, v0):
        self.robot = self.env.add_robot(robot())
        self.robot.reset_state(q0, v0)
    
    def get_state(self):
        """
        returns the current state of the robot
        """
        q, v = self.robot.get_state()
        return q, v
    
    def get_noisy_state(self):
        q, v = self.robot.get_noisy_state()
        return q, v        

    def send_joint_command(self, tau):
        """
        computes the torques using the ID controller and plugs the torques
        Input:
            tau : input torque
        """
        self.robot.send_joint_command(tau)
        self.env.step() # You can sleep here if you want to slow down the replay

    def get_current_contacts(self):
        """
        :return: an array of boolean 1/0 of end-effector current status of contact (0 = no contact, 1 = contact)
        """
        contact_configuration, _ = self.robot.end_effector_forces()
        return contact_configuration

    def get_ground_reaction_forces(self):
        """
        returns ground reaction forces from the simulator
        """
        _ , forces = self.robot.end_effector_forces()
        return forces
    
    def get_noisy_forces(self):
        forces = self.robot.get_noisy_force()
        return forces

    def start_recording(self, file_name):
        self.file_name = file_name
        self.logging_id  =pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, self.file_name)

    def stop_recording(self):
        pybullet.stopStateLogging(self.logging_id)
        
    def saveState(self):
        return pybullet.saveState()
        
    def restoreState(self, state_id):
        pybullet.restoreState(state_id)

    def get_imu_data(self):
        imu_gyro = self.robot.get_base_imu_angvel()
        imu_acc = self.robot.get_base_imu_linacc()
        imu_pos, imu_vel = self.robot.get_imu_frame_position_velocity()
        
        return imu_gyro, imu_acc, imu_pos, imu_vel
    
    def get_noisy_imu_data(self):
        imu_gyro = self.robot.get_noisy_base_imu_angvel()
        imu_acc = self.robot.get_noisy_base_imu_linacc()

        return imu_gyro, imu_acc
    
    def get_base_data(self):
        base_pos = self.robot.get_base_position_world()
        base_vel = self.robot.get_base_velocity_world()
        base_acc = self.robot.get_base_acceleration_world()
        return base_pos, base_vel, base_acc
    
    def setup_camera(self):
      
       num_links = pybullet.getNumJoints(self.robot.robot_id)

       for i in range(num_links):
            joint_info = pybullet.getJointInfo(self.robot.robot_id, i)
            if joint_info[1].decode() == self.camera_link_name:
                self.camera_link_index = joint_info[0]
                break
    
    def get_camera_link_index(self):
        return self.camera_link_index
       
    
    def capture_image(self, imange_number):
        link_state = pybullet.getLinkState(self.robot.robot_id, self.camera_link_index)
        cameraEye = link_state[0]
        forwardDir = [1, 0, 0]  # Forward direction
        cameraTarget = [cameraEye[0] + forwardDir[0],
                        cameraEye[1] + forwardDir[1],
                        cameraEye[2] + forwardDir[2]]
        cameraUp = [0, 0, 1]

        width = 640
        height = 480
        fov = 120
        aspect = width / height
        near = 0.02
        far = 15

        view_matrix = pybullet.computeViewMatrix(cameraEye, cameraTarget, cameraUp)
        projection_matrix = pybullet.computeProjectionMatrixFOV(fov, aspect, near, far)

        images = pybullet.getCameraImage(width,
                                         height,
                                         view_matrix,
                                         projection_matrix,
                                         renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        rgb_opengl = np.reshape(images[2], (height, width, 4)) * 1. / 255.

        

        return rgb_opengl  

    def save_image(self, image_number):

        image = self.capture_image(image_number)
        image = Image.fromarray((image * 255).astype(np.uint8))
        
        save_path = '/home/anudeep/devel/workspace/src/data/images'
        image.save(os.path.join(save_path, f"image_{image_number}.png"))
