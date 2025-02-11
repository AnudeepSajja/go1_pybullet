import numpy as np
from math import pi
from os.path import join, dirname
from os import environ
import pinocchio as se3
from pinocchio.utils import zero
from pinocchio.robot_wrapper import RobotWrapper
from robot_properties_go1.resources import Resources

class Go1Abstract(object):
    """ Abstract class used for all Solo robots. """

    # PID gains0
    kp = 12.0
    kd = 2.0
    ki = 0.0

    # Control time period.
    control_period = 0.001
    dt = control_period

    max_current = 40 # Ampers

    # Maximum torques.
    max_torque = 35 #Nm

    # Maximum control one can send, here the control is the current.
    max_control = max_torque

    # ctrl_manager_current_to_control_gain I am not sure what it does so 1.0.
    ctrl_manager_current_to_control_gain = 1.0

    max_qref = pi

    base_link_name = "trunk"

    rot_base_to_imu = np.identity(3)
    r_base_to_imu = np.zeros(3)

    @classmethod
    def buildRobotWrapper(cls):
        # Rebuild the robot wrapper instead of using the existing model to
        # also load the visuals.
        robot = RobotWrapper.BuildFromURDF(
            cls.urdf_path, cls.meshes_path, se3.JointModelFreeFlyer()
        )
        robot.model.rotorInertia[6:] = cls.motor_inertia
        robot.model.rotorGearRatio[6:] = cls.motor_gear_ration
        return robot

    def joint_name_in_single_string(self):
        joint_names = ""
        for name in self.robot_model.names[2:]:
            joint_names += name + " "
        return joint_names
    
class Go1Config(Go1Abstract):
    robot_name = "go1"
    
    resources = Resources(robot_name)
    meshes_path = resources.meshes_path
    urdf_path = resources.urdf_path
    ctrl_path = resources.imp_ctrl_yaml_path
    
    # The inertia of a single blmc_motor.
    motor_inertia = 0.000112

    # The motor gear ratio.
    motor_gear_ration = 1.0

    # pinocchio model.
    pin_robot_wrapper = RobotWrapper.BuildFromURDF(
        urdf_path, meshes_path, se3.JointModelFreeFlyer()
    )
    pin_robot_wrapper.model.rotorInertia[6:] = motor_inertia
    pin_robot_wrapper.model.rotorGearRatio[6:] = motor_gear_ration
    pin_robot = pin_robot_wrapper
    
    robot_model = pin_robot_wrapper.model
    #mass =  (32.3321 + 31.9058 + 37.2568 + 36.8306) / 9.81  #np.sum([i.mass for i in robot_model.inertias])
    # mass = (32.3321 + 31.9058 + 37.2568 + 36.8306) / 9.81 
    mass = 14.100428
    base_name = robot_model.frames[2].name

    # End effectors informations
    shoulder_ids = []
    end_eff_ids = []
    shoulder_names = []
    end_effector_names = []
    joint_names = []
    
    for leg in ["FL", "FR", "RL", "RR"]:
        end_eff_ids.append(robot_model.getFrameId(leg + "_foot_fixed"))
        end_effector_names.append(leg + "_foot_fixed")
        joint_names += [leg + "_hip_joint", leg + "_thigh_joint", leg + "_calf_joint"]
    
    nb_ee = len(end_effector_names)
    
    # The number of motors, here they are the same as there are only revolute joints.
    nb_joints = robot_model.nv - 6

    # Mapping between the ctrl vector in the device and the urdf indexes.
    urdf_to_dgm = tuple(range(12))

    map_joint_name_to_id = {}
    map_joint_limits = {}
    for i, (name, lb, ub) in enumerate(
        zip(
            robot_model.names[1:],
            robot_model.lowerPositionLimit,
            robot_model.upperPositionLimit,
        )
    ):
        map_joint_name_to_id[name] = i
        map_joint_limits[i] = [float(lb), float(ub)]

    # Define the initial state.
    # Joint limits: [Hip, Thigh, Calf] = [-1 to 1, -0.65 to 2.96, -2.72 to -0.83]
    initial_configuration = (
        [0.0, 0.0, 0.33, 0.0, 0.0, 0.0, 1.0]
        + 2 * [0.1, 0.7, -1.45]
        + 2 * [0.1, 0.7, -1.45]
    )
    initial_velocity = (8 + 4 + 6) * [
        0,
    ]

    q0 = zero(robot_model.nq)
    q0[:] = initial_configuration
    v0 = zero(robot_model.nv)
    a0 = zero(robot_model.nv)

    base_p_com = [0.0, 0.0, -0.05]

    rot_base_to_imu = np.identity(3)
    r_base_to_imu = np.array([-0.01592, -0.06659, -0.00617])
