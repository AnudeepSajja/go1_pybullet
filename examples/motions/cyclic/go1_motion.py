## Contains go1 gait params

import numpy as np
from motions.weight_abstract import BiconvexMotionParams
from robot_properties_go1.go1_config import Go1Config

pin_robot = Go1Config.buildRobotWrapper()
urdf_path = Go1Config.urdf_path

#### Stand #########################################
stand = BiconvexMotionParams("go1", "Stand")

# Cnt
stand.gait_period = 0.7
stand.stance_percent = [0.6, 0.6, 0.6, 0.6]
stand.gait_dt = 0.05
stand.phase_offset = [0.0, 0.5, 0.5, 0.0]

# IK
stand.state_wt = np.array([0., 0, 10] + [2500, 2500, 1500] + [4e2] * (12) \
                         + [0.00] * 3 + [100, 100, 100] + [2.75] *(12))

stand.ctrl_wt = [0, 0, 1000] + [5e2, 5e2, 5e2] + [5.5] *(12)

stand.swing_wt = [3*[1e4,], 3*[1e4,]]
stand.cent_wt = [3*[0*5e+2], 6*[2.75e+2]]  #CoM, Momentum
stand.step_ht = 0.15
stand.nom_ht = 0.27
stand.reg_wt = [5e-2, 1e-5]

# Dyn
stand.W_X =        np.array([3e-3, 3e-3, 1e+5, 5e+4, 5e+4, 2e+2, 1e+4, 1e+5, 1e4])
stand.W_X_ter = 10*np.array([1e+5, 1e+5, 5e+5, 5e+4, 5e+4, 2e+2, 1e+5, 1e+5, 1e5])
stand.W_F = np.array(4*[1e+1, 1e+1, 1e+1])
stand.rho = 5e+4
stand.ori_correction = [0.3, 0.7, 0.35]
stand.gait_horizon = 1.0
stand.kp = 75.0
stand.kd = 5.0
#

#### Trot #########################################
trot = BiconvexMotionParams("go1", "Trot")

# Cnt
trot.gait_period = 0.7
trot.stance_percent = [0.6, 0.6, 0.6, 0.6]
trot.gait_dt = 0.05
trot.phase_offset = [0.0, 0.5, 0.5, 0.0]

# IK
trot.state_wt = np.array([0., 0, 100] + [2500, 2500, 2500] + [4e2] * (12) \
                         + [0.00] * 3 + [100, 100, 100] + [2.75] *(12))

trot.ctrl_wt = [0, 0, 1000] + [5e2, 5e2, 5e2] + [5.5] *(12)

trot.swing_wt = [3*[1e4,], 3*[1e4,]]
trot.cent_wt = [3*[0*5e+1,], 6*[5e+2,]]  #CoM, Momentum
trot.step_ht = 0.06
trot.nom_ht = 0.27
trot.reg_wt = [5e-2, 1e-5]

# Dyn
trot.W_X = np.array([1e-5, 1e-5, 1e+5, 1e+1, 1e+1, 2e+2, 1e+4, 1e+4, 1e4])
trot.W_X_ter = 10*np.array([1e+5, 1e-5, 1e+5, 1e+1, 1e+1, 2e+2, 1e+5, 1e+5, 1e+5])
trot.W_F = np.array(4*[1e+1, 1e+1, 1e+1])
trot.rho = 5e+4
trot.ori_correction = [0.3, 0.7, 0.35]
trot.gait_horizon = 1.0
trot.kp = 75.0
trot.kd = 5.0
#


#### Walk #########################################
walk = BiconvexMotionParams("go1", "Walk")

# Gait Configuration (Cnt)
walk.gait_period = 0.8  # Slower than trot for a more stable walk
walk.stance_percent = [0.6, 0.6, 0.6, 0.6]  # Longer stance for stability
walk.gait_dt = 0.05  # Consistent with other motions
walk.phase_offset = [0.0, 0.5, 0.5, 0.0]  # Diagonal legs in stance

# Inverse Kinematics (IK) Parameters
walk.state_wt = np.array([0., 0, 10] + [1000, 1000, 1000] + [200] * 12 \
                         + [0.00] * 3 + [100, 100, 100] + [1.5] * 12)  # Lower weights than trot
walk.ctrl_wt = [0, 0, 500] + [300, 300, 300] + [2.0] * 12  # Lower control weights

# Swing and Centroidal Momentum Weights
walk.swing_wt = [3*[5e3,], 3*[5e3,]]  # Lower than trot for a more stable walk
walk.cent_wt = [3*[25,], 6*[150,]]  # Lower centroidal momentum weights

# Dynamic and Regulation Parameters
walk.step_ht = 0.05  # Lower step height for stability
walk.nom_ht = 0.27  # Consistent with GO1's nominal height
walk.reg_wt = [5e-2, 1e-5]  # Regularization weights

# Dynamics Weights (Dyn)
walk.W_X = np.array([1e-5, 1e-5, 2e+4, 5, 5, 100, 5e+3, 5e+3, 2e+3])  # Lower weights than trot
walk.W_X_ter = 10*np.array([1e-5, 1e-5, 2e+4, 5, 5, 100, 5e+3, 5e+3, 5e+3])  # Terrain adaptation
walk.W_F = np.array(4*[5, 5, 10])  # Lower force weights

# Stability and Orientation Correction
walk.rho = 2e+4  # Lower penalty for deviating from nominal height
walk.ori_correction = [0.2, 0.5, 0.3]  # Lower orientation correction

# Gait Horizon and Gains
walk.gait_horizon = 1.5  # Longer horizon for stability
walk.kp = 40.0  # Lower proportional gain than trot
walk.kd = 3.0  # Lower derivative gain than trot


#### Trot with Turning #########################################
trot_turn = BiconvexMotionParams("go1", "Trot_turn")

# Cnt
trot_turn.gait_period = 0.5
trot_turn.stance_percent = [0.6, 0.6, 0.6, 0.6]
trot_turn.gait_dt = 0.05
trot_turn.phase_offset = [0.0, 0.4, 0.4, 0.0]

# IK
trot_turn.state_wt = np.array([0., 0, 10] + [500, 500, 10] + [1.0] * (pin_robot.model.nv - 6) \
                              + [0.00] * 3 + [100, 100, 10] + [0.5] *(pin_robot.model.nv - 6))

trot_turn.ctrl_wt = [0, 0, 1000] + [5e2, 5e2, 5e2] + [1.0] *(pin_robot.model.nv - 6)

trot_turn.swing_wt = [1e4, 1e4]
trot_turn.cent_wt = [0*5e+1, 5e+2]
trot_turn.step_ht = 0.05
trot_turn.nom_ht = 0.2
trot_turn.reg_wt = [5e-2, 1e-5]

# Dyn
trot_turn.W_X =        np.array([1e-5, 1e-5, 1e+5, 1e+1, 1e+1, 2e+2, 1e+4, 1e+4, 1e4])
trot_turn.W_X_ter = 10*np.array([1e+5, 1e-5, 1e+5, 1e+1, 1e+1, 2e+2, 1e+5, 1e+5, 1e+5])
trot_turn.W_F = np.array(4*[1e+1, 1e+1, 1e+1])
trot_turn.rho = 5e+4
trot_turn.ori_correction = [0.0, 0.5, 0.4]
trot_turn.gait_horizon = 1.0
trot_turn.kp = 3.0
trot_turn.kd = 0.05

#### Jump #########################################
jump = BiconvexMotionParams("go1", "Jump")

# Cnt
jump.gait_period = 0.5
jump.stance_percent = [0.4, 0.4, 0.4, 0.4]
jump.gait_dt = 0.05
jump.phase_offset = [0.3, 0.3, 0.3, 0.3]

# IK
jump.state_wt = np.array([0., 0, 10] + [1500, 1500, 1500] + [5e2] * (pin_robot.model.nv - 6) \
                        + [0.00] * 3 + [100, 100, 100] + [2.75] *(pin_robot.model.nv - 6))

jump.ctrl_wt = [0, 0, 1000] + [5e2, 5e2, 5e2] + [10.0] *(pin_robot.model.nv - 6)

jump.swing_wt = [1e4, 1e4]
jump.cent_wt = [3*[0*5e+1,], 6*[2.75e+2,]]
jump.step_ht = 0.05
jump.nom_ht = 0.27
jump.reg_wt = [5e-2, 1e-5]

# Dyn 
jump.W_X =        np.array([1e-5, 1e-5, 1e+5, 1e+1, 1e+1, 2e+2, 1e+4, 1e+4, 1e4])
jump.W_X_ter = 10*np.array([1e+5, 1e-5, 1e+5, 1e+1, 1e+1, 2e+2, 1e+5, 1e+5, 1e+5])
jump.W_F = np.array(4*[1e+2, 1e+2, 1e+2])
jump.rho = 5e+4
jump.ori_correction = [0.35, 0.5, 0.45]
jump.gait_horizon = 1.5
jump.kp = 100.0
jump.kd = 5.0


#### Bound #########################################
bound = BiconvexMotionParams("go1", "Bound")

# Gait Configuration (Cnt)
bound.gait_period = 0.4  # Between trot and jump settings, considering dynamic motion
bound.stance_percent = [0.5, 0.5, 0.5, 0.5]  # Symmetrical for bound motion
bound.gait_dt = 0.05  # Consistent with other motions
bound.phase_offset = [0.0, 0.0, 0.5, 0.5]  # Alternating legs for bound

# Inverse Kinematics (IK) Parameters
bound.state_wt = np.array([0., 0, 100] + [2000, 2000, 2000] + [300] * 12 \
                         + [0.00] * 3 + [100, 100, 100] + [2.0] * 12)  # Adjusted based on trot/jump
bound.ctrl_wt = [0.5, 0.5, 1000] + [500, 500, 500] + [3.0] * 12  # Adjusted control weights

# Swing and Centroidal Momentum Weights
bound.swing_wt = [3*[1e4,], 3*[1e4,]]  # High for dynamic motion
bound.cent_wt = [3*[50,], 6*[250,]]  # Adjusted based on GO1 trot/jump

# Dynamic and Regulation Parameters
bound.step_ht = 0.07  # Similar to Solo12 for dynamic motion
bound.nom_ht = 0.27  # Consistent with GO1's jump height
bound.reg_wt = [5e-2, 1e-5]  # Regularization weights

# Dynamics Weights (Dyn)
bound.W_X = np.array([1e-5, 1e-5, 5e+4, 10, 10, 200, 1e+4, 1e+4, 5e+3])  # Adjusted from Solo12
bound.W_X_ter = 10*np.array([1e-5, 1e-5, 5e+4, 10, 10, 200, 1e+4, 1e+4, 1e+4])  # Terrain adaptation
bound.W_F = np.array(4*[10, 10, 15])  # Force weights

# Stability and Orientation Correction
bound.rho = 5e+4  # Penalty for deviating from nominal height
bound.ori_correction = [0.3, 0.7, 0.5]  # Adjusted based on trot/jump

# Gait Horizon and Gains
bound.gait_horizon = 1.2  # Intermediate horizon
bound.kp = 60.0  # Proportional gain, higher than Solo12 but aligned with GO1 trot/jump
bound.kd = 5.0  # Derivative gain, consistent with GO1 jump
