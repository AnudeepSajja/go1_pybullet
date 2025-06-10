import numpy as np
import pybullet

from bullet_utils.wrapper import PinBulletWrapper
from robot_properties_go1.go1_config import Go1Config

dt = 1e-3

class Go1Robot(PinBulletWrapper):
    def __init__(
        self,
        pos=None,
        orn=None,
        useFixedBase=False,
    ):

        # Load the robot
        if pos is None:
            pos = [0.0, 0, 0.45]
        if orn is None:
            orn = pybullet.getQuaternionFromEuler([0, 0, 0])

        pybullet.setAdditionalSearchPath(Go1Config.resources.package_path)
        self.urdf_path = Go1Config.urdf_path
        self.robotId = pybullet.loadURDF(
            self.urdf_path,
            pos,
            orn,
            flags=pybullet.URDF_USE_INERTIA_FROM_FILE,
            useFixedBase=useFixedBase,
        )
        pybullet.getBasePositionAndOrientation(self.robotId)

        # Create the robot wrapper in pinocchio.
        self.pin_robot = Go1Config.buildRobotWrapper()

        # Query all the joints.
        num_joints = pybullet.getNumJoints(self.robotId)

        for ji in range(num_joints):
            pybullet.changeDynamics(
                self.robotId,
                ji,
                linearDamping=0.01,
                angularDamping=0.01,
                # restitution=0.0,
                lateralFriction=0.2,
            )

        self.base_link_name = "trunk"
        self.end_eff_ids = []
        self.end_effector_names = []
        self.joint_names = []

        for leg in ["FL", "FR", "RL", "RR"]:
            self.joint_names += [leg + "_hip_joint", leg + "_thigh_joint", leg + "_calf_joint"]
            self.end_eff_ids.append(
                self.pin_robot.model.getFrameId(leg + "_foot_fixed")
            )
            self.end_effector_names.append(leg + "_foot_fixed")

        self.nb_ee = len(self.end_effector_names)
        
        # Creates the wrapper by calling the super.__init__.
        super(Go1Robot, self).__init__(
            self.robotId,
            self.pin_robot,
            self.joint_names,
            ["FL_foot_fixed", "FR_foot_fixed", "RL_foot_fixed", "RR_foot_fixed"],
            Go1Config,
        )

    def forward_robot(self, q=None, dq=None):
        if not q:
            q, dq = self.get_state()
        elif not dq:
            raise ValueError("Need to provide q and dq or non of them.")

        self.pin_robot.forwardKinematics(q, dq)
        self.pin_robot.computeJointJacobians(q)
        self.pin_robot.framesForwardKinematics(q)
        self.pin_robot.centroidalMomentum(q, dq)

    def reset_to_initial_state(self) -> None:
        """Reset robot state to the initial configuration (based on Solo12Config)."""
        q0 = np.matrix(Go1Config.initial_configuration).T
        dq0 = np.matrix(Go1Config.initial_velocity).T
        self.reset_state(q0, dq0)
    