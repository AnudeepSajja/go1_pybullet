import os
import time
import numpy as np
import torch
import pandas as pd
import pinocchio as pin
from END2ENDPredictor import NMPCPredictor
from robot_properties_go1.go1_wrapper import Go1Robot, Go1Config
from envs.pybullet_env import PyBulletEnv


# utils.py

import numpy as np
import pandas as pd
import torch
from END2ENDPredictor import NMPCPredictor


def load_model_and_normalizer(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    neurons = 2560
    input_size = checkpoint['model_state_dict']['hidden1.weight'].shape[1]
    output_size = checkpoint['model_state_dict']['output.weight'].shape[0]

    model = NMPCPredictor(input_size=input_size, output_size=output_size, neurons=neurons).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    X_min = np.array(checkpoint['x_min'])
    X_max = np.array(checkpoint['x_max'])
    Y_min = np.array(checkpoint['y_min'])
    Y_max = np.array(checkpoint['y_max'])

    return model, X_min, X_max, Y_min, Y_max


def preprocess_eval_data(data_path, eval_steps, X_min, X_max):
    df = pd.read_csv(data_path).head(eval_steps)

    imu_cols = [col for col in df.columns if col.startswith('imu_acc_') or col.startswith('imu_gyro_')]
    qj_cols = [col for col in df.columns if col.startswith('qj_')]
    dqj_cols = [col for col in df.columns if col.startswith('dqj_')]
    foot_cols = [col for col in df.columns if col.startswith('foot_')]

    X_all = df[imu_cols + qj_cols + dqj_cols + foot_cols]
    assert X_all.shape[1] == 34, f"Expected 34 features, got {X_all.shape[1]}"

    X_to_normalize = X_all[imu_cols + qj_cols + dqj_cols]
    X_foot = X_all[foot_cols]
    X_norm = 2 * (X_to_normalize - X_min) / (X_max - X_min + 1e-8) - 1
    X_input = pd.concat([X_norm, X_foot], axis=1)

    return torch.tensor(X_input.values, dtype=torch.float32)


def predict_joint_positions(model, X_tensor, Y_min, Y_max):
    with torch.no_grad():
        pred_q_norm = model(X_tensor)
    pred_q = ((pred_q_norm.cpu().numpy() + 1) / 2) * (Y_max - Y_min) + Y_min
    return pred_q
