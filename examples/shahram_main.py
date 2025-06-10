#!/usr/bin/env python

import os
import copy
import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from envs.pybullet_terrain_env import PyBulletTerrainEnv
from robot_properties_go1.go1_wrapper import Go1Robot, Go1Config
from shahram_utils import Replay_buff_vision
from shahram_td3 import TD3

MODEL_PATH = "/home/anudeep/devel/workspace/data_humanoids/models/stl_humaoinds_trot_256n_new.pth"

# === Hyperparameters ===
BATCH_SIZE = 256
REPLAY_INITIAL = 256
MAX_STEPS = 1000
TOTAL_TRAIN_STEPS = int(1e6)
KP = 75.0
KD = 5.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(log_dir=os.path.join(os.path.dirname(__file__), 'runs', 'td3_' + datetime.datetime.now().strftime("%d.%h.%H.%M")))

def rgb_to_grayscale(image):
    """Convert an RGB image to grayscale."""
    return np.dot(image[..., :3], [0.299, 0.587, 0.114])

if __name__ == "__main__":
    print("Initializing robot...")
    robot_config = Go1Config()
    q0 = np.array(robot_config.initial_configuration)
    v0 = np.zeros(robot_config.buildRobotWrapper().model.nv)
    env = PyBulletTerrainEnv(Go1Robot, q0, v0)

    print("Resetting environment...")
    state_sample, img_sample = env.reset()
    STATE_DIMENSION = state_sample.shape[0]
    ACTION_DIMENSION = 12
    img_sample_gray = rgb_to_grayscale(img_sample)
    fake_img = torch.from_numpy(img_sample_gray).float().to(device)
    if fake_img.ndim == 2:
        fake_img = fake_img.unsqueeze(0)  # Corrected: Only add channel dimension (C, H, W)

    print("Initializing TD3 policy...")
    policy = TD3(STATE_DIMENSION, ACTION_DIMENSION, fake_img.shape)

    # Load imitation checkpoint
    print("Loading pretrained model...")
    ckpt = torch.load(MODEL_PATH, map_location=device)
    policy.imit_net.load_state_dict(ckpt['model_state_dict'])
    X_min = ckpt['x_min'].astype(np.float32)
    X_max = ckpt['x_max'].astype(np.float32)
    Y_min = ckpt['y_min'].astype(np.float32)
    Y_max = ckpt['y_max'].astype(np.float32)

    print("Transferring imitation weights to actor...")
    policy.transfer_imit_to_actor()
    policy.actor_tgt.replicate()
    policy.critic_tgt.replicate()

    replay_buffer = Replay_buff_vision(STATE_DIMENSION, ACTION_DIMENSION, fake_img.shape, BATCH_SIZE, device=device)

    tot_steps, train_steps, episode = 0, 0, 0
    print("Starting training...")

    while train_steps < TOTAL_TRAIN_STEPS:
        episode += 1
        print(f"Starting Episode {episode}...")
        state, img = env.reset()
        img_gray = rgb_to_grayscale(img)
        ep_reward = 0.0

        for step in range(MAX_STEPS):
            tot_steps += 1

            # Normalize state
            x_raw = state[:-4]
            contacts = state[-4:]
            x_norm = 2.0 * (x_raw - X_min) / (X_max - X_min) - 1.0
            state_norm = np.concatenate([x_norm, contacts], axis=0)

            if len(replay_buffer) < REPLAY_INITIAL:
                # Random action if replay buffer is not initialized
                target_q = np.random.uniform(-1, 1, ACTION_DIMENSION)
            else:
                img_tensor = torch.tensor(img_gray).float().to(device)
                if img_tensor.ndim == 2:
                    img_tensor = img_tensor.unsqueeze(0)  # Ensure correct shape (C, H, W)
                target_q = policy.select_action(state_norm, img_tensor)

            cur_q, cur_v = env.get_state()
            
            # Denormalize the target_q using Y_min and Y_max
            Y_max = np.array(Y_max)
            Y_min = np.array(Y_min)
            denorm_target_q = ((target_q + 1) / 2) * (Y_max.reshape(1, -1) - Y_min.reshape(1, -1)) + Y_min.reshape(1, -1)
            
            # Debugging print statements
            # print(f"Step {step}: Target Q (normalized): {target_q}")
            # print(f"Step {step}: Target Q (denormalized): {denorm_target_q}")
            
            cur_motor_pos = np.array(cur_q[7:])
            cur_motors_vel = np.array(cur_v[6:])
            
            # Compute torque using PD control
            tau = KP * (denorm_target_q - cur_motor_pos) + KD * (0.0 - cur_motors_vel)
            tau = tau.flatten()
            
            # Debugging print statement for torque
            # print(f"Step {step}: Torque: {tau}")

            (next_state, next_img), reward, done, _ = env.step(tau)
            next_img_gray = rgb_to_grayscale(next_img)
            ep_reward += reward

            next_state = next_state[:STATE_DIMENSION]
            
            # Add experience to replay buffer
            replay_buffer.add(state, target_q, next_state, reward, float(done), img_gray, next_img_gray)

            # Update state and image for the next step
            state = copy.deepcopy(next_state)
            img_gray = copy.deepcopy(next_img_gray)

            # Train the policy if replay buffer is ready
            if len(replay_buffer) >= REPLAY_INITIAL:
                policy.train(replay_buffer, BATCH_SIZE)
                train_steps += 1
                if train_steps % 1000 == 0:
                    print(f"[Episode {episode}] Train step: {train_steps}, Ep reward: {ep_reward:.2f}")

            # Log metrics to TensorBoard
            writer.add_scalar('Episode/Reward', ep_reward, episode)
            writer.add_scalar('Train/TotalSteps', tot_steps, tot_steps)

            if done or step == MAX_STEPS - 1:
                print(f"Episode {episode} done. Reward: {ep_reward:.2f}")
                break

    writer.close()
    print("Training complete.")