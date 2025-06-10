import numpy as np 
import torch 
import copy
import cv2
import os


class Replay_buff_vision(object):
    def __init__(self, state_dim, action_dim, img_dim, batch_size, device, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        
        # Image paths and directories
        self.img_path = "/home/anudeep/devel/workspace/data_humanoids/train_img_path"
        self.next_img_path = "/home/anudeep/devel/workspace/data_humanoids/train_next_img_path"
        os.makedirs(self.img_path, exist_ok=True)
        os.makedirs(self.next_img_path, exist_ok=True)
        
        # Configuration parameters
        self.img_dim = img_dim  # Expected image shape (e.g., (1, 84, 84))
        self.batch_size = batch_size
        self.device = device
    
    def __len__(self):  # Recommended Pythonic approach
        return self.size


    def add(self, state, action, next_state, reward, done, img, next_img):
        # Validate input images
        if img is None or next_img is None:
            raise ValueError("Cannot add None images to buffer")
            
        # Store transition data
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1 - done
        
        # Save images to disk
        self.save_images(img, next_img)
        
        # Update pointers
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def save_images(self, img, nxt_img):
        """Save images to disk with proper scaling and validation"""
        # Validate input range
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
            nxt_img = (nxt_img * 255).astype(np.uint8)
            
        img_path = f"{self.img_path}/image_{self.ptr}.png"
        next_img_path = f"{self.next_img_path}/next_image_{self.ptr}.png"
        
        # Save with existence check
        if not cv2.imwrite(img_path, img):
            raise RuntimeError(f"Failed to save image at {img_path}")
        if not cv2.imwrite(next_img_path, nxt_img):
            raise RuntimeError(f"Failed to save image at {next_img_path}")

    def load_images(self, index):
        """Load images with robust error handling"""
        img_path = f"{self.img_path}/image_{index}.png"
        next_img_path = f"{self.next_img_path}/next_image_{index}.png"
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        next_img = cv2.imread(next_img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or next_img is None:
            raise FileNotFoundError(
                f"Missing image at index {index}. "
                f"Checked paths: {img_path} and {next_img_path}"
            )
            
        # Normalize and add channel dimension
        return (
            img.astype(np.float32)[None, ...] / 255.0,  # Shape: (1, H, W)
            next_img.astype(np.float32)[None, ...] / 255.0
        )

    def sample(self):
        """Sample batch with proper tensor conversion"""
        ind = np.random.randint(0, self.size, size=self.batch_size)
        
        # Load images directly into tensors
        imgs, next_imgs = [], []
        for idx in ind:
            img, next_img = self.load_images(idx)
            imgs.append(torch.from_numpy(img))
            next_imgs.append(torch.from_numpy(next_img))
        
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.stack(imgs).float().to(self.device),
            torch.stack(next_imgs).float().to(self.device)
        )
class TargetNet:
    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def alpha_sync(self, alpha):
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k,v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1- alpha) * v
        self.target_model.load_state_dict(tgt_state)
        
    def replicate(self):
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k,v in state.items():
            tgt_state[k] = copy.deepcopy(v)
        self.target_model.load_state_dict(tgt_state)

