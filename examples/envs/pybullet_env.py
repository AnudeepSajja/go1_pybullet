## this file contains a simulation env with pybullet
## Author : Avadesh Meduri
## Date : 7/05/2021

import pybullet
from bullet_utils.env import BulletEnvWithGround

class PyBulletEnv:

    def __init__(self, robot, q0, v0):

        print("loading bullet")
        self.env = BulletEnvWithGround()
        self.robot = self.env.add_robot(robot())
        self.robot.reset_state(q0, v0)
        # pybullet.resetDebugVisualizerCamera( cameraDistance=-1.5, cameraYaw=45, cameraPitch=-10, cameraTargetPosition=[0,0,.8])
        ## For data recording
        self.q_arr = []
        self.v_arr = []
    
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

