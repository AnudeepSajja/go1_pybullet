#!/usr/bin/env python

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import os
import numpy as np
import sys 
import copy
from shahram_utils import TargetNet



OUT_FEATURES = 20
MATCH_WEIGHT = 0.2

expl_noise = 0.1
GAMMA = 0.99
REPLAY_INITIAL = 4000
MAX_STEPS = 1000
LR_ACTS = 3e-4
LR_VALS = 3e-4
policy_noise=0.2
noise_clip=0.5
policy_freq=2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#---Directory Path---#
dirPath = os.path.dirname(os.path.realpath(__file__))


class Imit_Net(nn.Module):
    def __init__(self, input_size, output_size, neurons=256):  # Add neurons as a parameter
        super(Imit_Net, self).__init__()
        self.neurons = neurons  # Store the neurons value
        self.hidden1 = nn.Linear(input_size, self.neurons)
        self.act1 = nn.ELU()
        self.hidden2 = nn.Linear(self.neurons, self.neurons)
        self.act2 = nn.ELU()
        self.hidden3 = nn.Linear(self.neurons, self.neurons)
        self.act3 = nn.ELU()
        self.output = nn.Linear(self.neurons, output_size)

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        x = self.output(x)
        return x


class Actor(nn.Module):
    def init_wts(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0.01)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0.01)



    def __init__(self, obs_size, action_size, image_size):
        super(Actor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(image_size[0], 16, kernel_size=4, stride=2, padding =1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding =1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding =1),
            nn.BatchNorm2d(16),
            nn.ReLU()
            )
        conv_out_size = self._get_conv_out(image_size)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 64),
            nn.ReLU(),
            nn.Linear(64, OUT_FEATURES)
            )

        self.act_net = nn.Sequential(
            nn.Linear(obs_size + OUT_FEATURES, 256),
            nn.ELU(), 
            nn.Linear(256,256), 
            nn.ELU(), 
            nn.Linear(256,256), 
            nn.ELU(), 
            nn.Linear(256,action_size)
        )

        self.apply(self.init_wts)
        self.act_net.apply(self.init_wts)
        self.fc.apply(self.init_wts)
        self.conv.apply(self.init_wts)

        
    def _get_conv_out(self, shape):
        o = F.adaptive_avg_pool2d(self.conv(torch.zeros(1, *shape)),1)
        return int(np.prod(o.size()))

    def conv_forward(self, img):
        conv_out = F.adaptive_avg_pool2d(self.conv(img),1).view(img.size()[0], -1)
        return self.fc(conv_out)

    def forward(self, x, img):
        img_x = self.conv_forward(img)
        stat_tot = torch.cat([x,img_x],1)
        action = self.act_net(stat_tot)
        return  torch.tanh(action) 

class Critic(nn.Module):
    def init_wts(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0.01)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0.01)

    def __init__(self, obs_size, action_size, image_size):
        super(Critic,self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(image_size[0], 16, kernel_size=3, stride=2, padding =1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding =1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding =1),
            nn.BatchNorm2d(16),
            nn.ReLU()
            )

        conv_out_size = self._get_conv_out(image_size)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 64),
            nn.ReLU(),
            nn.Linear(64, OUT_FEATURES)
            )
        self.q1_val = nn.Sequential(
            nn.Linear(obs_size + OUT_FEATURES + action_size, 256), 
            nn.ReLU(),
            nn.Linear(256, 256), 
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.q2_val = nn.Sequential(
            nn.Linear(obs_size + OUT_FEATURES + action_size, 256), 
            nn.ReLU(),
            nn.Linear(256, 256), 
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.apply(self.init_wts)
        self.q1_val.apply(self.init_wts)
        self.q2_val.apply(self.init_wts)
        self.fc.apply(self.init_wts)
        self.conv.apply(self.init_wts)

    def _get_conv_out(self, shape):
        o = F.adaptive_avg_pool2d(self.conv(torch.zeros(1, *shape)),1)
        return int(np.prod(o.size()))
        
    def conv_forward(self, img):
        conv_out = F.adaptive_avg_pool2d(self.conv(img),1).view(img.size()[0], -1)
        return self.fc(conv_out)

    def forward(self, x, a, img):
        img_x = self.conv_forward(img)
        stat_x = torch.cat([x,img_x],1)
        state_act = torch.cat([stat_x,a],1)
        q_1 = self.q1_val(state_act)
        q_2 = self.q2_val(state_act)
        return q_1,q_2

    def q1 (self, x, a, img):
        img_x = self.conv_forward(img)
        stat_x = torch.cat([x,img_x],1)
        state_act = torch.cat([stat_x,a],1)
        q_1 = self.q1_val(state_act)
        return q_1

class TD3(object):
    def __init__(self, state_dim, action_dim, img_dim, discount = 0.99,
    policy_noise_v=0.2, policy_noise_w=0.2, noise_clip_v=0.5, noise_clip_w=0.5, policy_freq=2):
        
        self.actor = Actor(state_dim, action_dim, img_dim).to(device)
        self.actor_tgt = TargetNet(self.actor)
        self.actor_optim = optim.Adam(self.actor.parameters(), LR_ACTS)

        self.critic = Critic(state_dim, action_dim, img_dim).to(device)
        self.critic_tgt = TargetNet(self.critic)
        self.critic_optim = optim.Adam(self.critic.parameters(),LR_VALS)

        self.imit_net = Imit_Net(state_dim, action_dim).to(device)

        self.discount = discount
        self.policy_noise_v = policy_noise_v
        self.policy_noise_w = policy_noise_w
        self.noise_clip_v = noise_clip_v
        self.noise_clip_w = noise_clip_w
        self.policy_freq = policy_freq

        self.obs_size = state_dim
        self.action_dim = action_dim

        self.total_it = 0

    def select_action(self,state, img):
        # state = torch.from_numpy(state.reshape(1,-1)).to(device)
        
        state = torch.from_numpy(state.reshape(1, -1)).float().to(device)

        # img_v = torch.tensor(np.array([np.reshape(img, [1,84, 84])], copy=False)).to(device)
        
        # Ensure img is a tensor and reshape to [batch=1, channels=1, height=84, width=84]
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img).float().to(device)
        img_v = img.view(1, 1, 84, 84)  # Reshape to (1, 1, 84, 84)

        action = self.actor(state, img_v).cpu().detach().numpy()[0]
        action += np.random.normal(scale = expl_noise, size = self.action_dim)
        action = np.clip(action, -1, 1)
        return action

    def train(self,replay_buffer, batch_size = 256):
        self.total_it +=1
         
        state, action, next_state, reward, not_done, img, next_img = replay_buffer.sample()

        ##########################################################################################
        
        with torch.no_grad():
            torch.autograd.set_detect_anomaly(True)
            next_action = self.actor_tgt.target_model(next_state, next_img)
            noise = torch.normal(0, 0.2, size=next_action.shape).to(state.device)
            noise = noise.clamp(-0.5, 0.5)
            next_action = (next_action + noise).clamp(-1, 1)
            next_action = torch.clamp(next_action,-1, 1)

            action_gait = self.imit_net(state)
            
        #########################################################################################
        
        # with torch.no_grad():
        #     torch.autograd.set_detect_anomaly(True)
        #     next_action = self.actor_tgt.target_model(next_state, next_img)
        #     next_action += torch.clamp(torch.tensor(np.random.normal(scale = 0.2)),-0.5,0.5)
        #     next_action = torch.clamp(next_action,-1, 1)

        #     action_gait = self.imit_net(state)

        ############################################################################################

        
        # compute target Q value
        tgt_q1, tgt_q2 = self.critic_tgt.target_model(next_state, next_action, next_img)
        tgt_q = torch.min(tgt_q1,tgt_q2)
        tgt_q = reward + not_done * self.discount*tgt_q
    # critic updates **********************************************
        curr_q1, curr_q2 = self.critic(state, action, img)
        
        crt_loss = F.mse_loss(curr_q1,tgt_q) + F.mse_loss(curr_q2, tgt_q) + MATCH_WEIGHT *F.mse_loss(action_gait, self.actor(state,img).detach())

        self.critic_optim.zero_grad()
        crt_loss.backward()
        self.critic_optim.step()
    # delayed updates********************************************** 
        if self.total_it % self.policy_freq == 0:
            act_loss = -self.critic.q1(state, self.actor(state,img), img) + MATCH_WEIGHT *F.mse_loss(self.actor(state,img), action_gait.detach())
            act_loss = act_loss.mean()

            self.actor_optim.zero_grad()
            act_loss.backward()
            self.actor_optim.step()

            self.actor_tgt.alpha_sync(alpha=1 - 0.005)
            self.critic_tgt.alpha_sync(alpha=1 - 0.005)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optim.state_dict(), filename + "_critic_optimizer")
        torch.save(self.imit_net.state_dict(), filename + "_imit_net")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optim.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optim.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_tgt = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optim.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_tgt = copy.deepcopy(self.actor)

    def copy_imit(self, filename):
        # use this function to copy the weights from imit_net to the initialized imit_net
        ckpt = torch.load(filename)
        self.imit_net.load_state_dict(ckpt["model_state_dict"])
        
    def transfer_imit_to_actor(self):
        """
        Transfer weights from imitation net (fc) to actor net (act_net),
        padding the first layer to match the input dimension difference.
        """
        # ----------------------
        # First layer: padded copy
        # ----------------------
        fc_layer0 = self.imit_net.hidden1        # nn.Linear(obs_size, 256)
        act_layer0 = self.actor.act_net[0]  # nn.Linear(obs_size + OUT_FEATURES, 256)

        assert act_layer0.in_features == self.obs_size + OUT_FEATURES
        # assert act_layer0.out_features == OUT_FEATURES

        # Pad the weights with zeros on extra input dims
        weight_fc = fc_layer0.weight.data  # shape: (256, obs_size)
        bias_fc = fc_layer0.bias.data      # shape: (256,)
        pad_dim = act_layer0.in_features - weight_fc.shape[1]

        padded_weight = torch.cat([weight_fc, torch.zeros(weight_fc.size(0), pad_dim)], dim=1)

        with torch.no_grad():
            act_layer0.weight.copy_(padded_weight)
            act_layer0.bias.copy_(bias_fc)

        # ----------------------
        # Second hidden layer: copy directly
        # ----------------------
        with torch.no_grad():   
            # print(self.imit_net[2])
            self.actor.act_net[2].weight.copy_(self.imit_net.hidden2.weight)
            self.actor.act_net[2].bias.copy_(self.imit_net.hidden2.bias)
        # ----------------------
        # Third hidden layer: copy directly
        # ----------------------
        
        with torch.no_grad():
            self.actor.act_net[4].weight.copy_(self.imit_net.hidden3.weight)
            self.actor.act_net[4].bias.copy_(self.imit_net.hidden3.bias)
        # ----------------------
        # Output layer: copy directly
        # ----------------------
        with torch.no_grad():
            self.actor.act_net[6].weight.copy_(self.imit_net.output.weight)
            self.actor.act_net[6].bias.copy_(self.imit_net.output.bias)

    # Example usage:
    # transfer_fc_to_actnet(self.act_net, obs_size, OUT_FEATURES)



   