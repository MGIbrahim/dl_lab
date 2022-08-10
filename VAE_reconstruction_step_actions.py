#from pl_bolts.datamodules import CIFAR10DataModule
from lib2to3.pgen2.literals import simple_escapes
from pl_bolts.models.autoencoders import VAE
from pytorch_lightning import Trainer
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import random
from torchvision.models import resnet18 
import torchvision.transforms as T
from PIL import Image
from custom_VAE import custom_VAE
import cv2


class CustomDataset(Dataset):
    def __init__(self, batch_rgb_static_last_obs, batch_rgb_static_first_obs, rgb_static_tensor, actions):
        self.batch_rgb_static_last_obs = batch_rgb_static_last_obs
        self.batch_rgb_static_first_obs = batch_rgb_static_first_obs
        self.rgb_static_tensor = rgb_static_tensor
        self.actions = actions
    
    def __len__(self):
        return self.batch_rgb_static_last_obs.shape[0]
        
    def __getitem__(self, index):
        return self.batch_rgb_static_last_obs[index], self.batch_rgb_static_first_obs[index], self.rgb_static_tensor[index], self.actions[index]

#Loading GTI demonstrations
def read_data(path):
    indices = [141,3716] #filtered gti_demos
    indices = list(range(indices[0], indices[1] + 1))
    data = ['rgb_static', 'rgb_gripper']

    idx = indices[0]
    i = 0
    len_indices = indices[-1] - indices[0]
    rgb_static = [0] * (len_indices + 1)
    rgb_gripper = [0] * (len_indices + 1)
    actions = [0] * (len_indices + 1)
    rel_actions = [0] * (len_indices + 1)
    robot_obs = [0] * (len_indices + 1)
    scene_obs = [0] * (len_indices + 1)
    for idx in indices:
        t = np.load(f'{path}/episode_{idx:07d}.npz', allow_pickle=True)
        print(f"episode_{indices[i]:07d}.npz")
        for d in data:
            if d == 'rgb_static':
                rgb_static[i]  = t[d][:,:,::-1] # Converts from BGR to RGB
            elif d == 'rgb_gripper':
                rgb_gripper[i]  = t[d][:,:,::-1] # Converts from BGR to RGB

        actions[i]  = t['actions']
        rel_actions[i]  = t['rel_actions']
        robot_obs[i] = t['robot_obs']
        scene_obs[i] = t['scene_obs']
        i+=1
    
    return np.array(rgb_static), np.array(rgb_gripper), np.array(actions), np.array(robot_obs)
 
def random_sampler(rgb_static, actions, robot_obs, H):
    # H is the sample length
    #indices = [141,3716], len = 3576
    # range of indices starts from 141 to 3716 - (H * 10) as steps of 10 are taken
    # and to prevent getting into index out of bounds error 
    #indices = list(range(len(rgb_static) - H * 10))
    indices = np.array([2723, 1081, 141, 2042, 912, 364, 469, 795, 637, 1362, 2564, 1524, 2225, 3356, 2417, 2890, 3030, 3176, 1025]) - 141

    random_indices = random.choices(indices, k = len(rgb_static))

    rgb_static_tensor = torch.zeros((len(rgb_static), H, 3, rgb_static.shape[2], rgb_static.shape[3]), dtype=torch.uint8)
    actions_tensor = torch.zeros((len(rgb_static), H, actions.shape[1]), dtype=torch.float64)
    robot_obs_tensor = torch.zeros((len(rgb_static), H, robot_obs.shape[1],), dtype=torch.float64)

    i = 0
    for index in random_indices:
        rgb_static_tensor[i] = rgb_static[index : index + H * 10 : 10]
        actions_tensor[i] = torch.from_numpy(actions[index + 9 : index + H * 10 : 10]) # actions[0] --> robot_obs[1]
        robot_obs_tensor[i] = torch.from_numpy(robot_obs[index : index + H * 10 : 10])
        i+=1

    return rgb_static_tensor, actions_tensor, robot_obs_tensor 

def rgb2gray(rgb):
    """ 
    this method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721])
    return gray.astype('float32') 

def resize(rgb_static):
    """ 
    Resizes rgb_static images from (200,200) to (32,32)
    """

    rgb_static_tensor_resized = torch.zeros((len(rgb_static), 3, 32, 32), dtype=torch.uint8)
    
    transform = T.Compose([
        T.Resize(size = (32, 32)),
        T.PILToTensor()
    ])

    for i in range(len(rgb_static)):

        img = Image.fromarray(rgb_static[i][:,:,::-1])

        rgb_static_tensor_resized[i] = transform(img)
    return rgb_static_tensor_resized

def intersection(rgb_static, robot_obs):
    H  = 15

    task_a = [141 - 141, 469 - 141, 637 - 141, 1362 - 141, 2225 - 141, 2417 - 141, 2890 - 141, 3030 - 141, 3176 - 141] # block in drawer
    task_b = [364 - 141, 795 - 141, 912 - 141, 1081 - 141, 1025 - 141, 2042 - 141, 2564 - 141, 2723 - 141, 3356 - 141] # block on table

    rgb_static_seq_a = np.zeros((len(task_a), H, 3, 32, 32), dtype=np.uint8)
    rgb_static_seq_b = np.zeros((len(task_b), H, 3, 32, 32), dtype=np.uint8)

    robot_obs_seq_a = np.zeros((len(task_a), H, 6))
    robot_obs_seq_b = np.zeros((len(task_b), H, 6))

    similarity = []

    # i : Task Index, k : 
    for i in range(len(task_a)):
        rgb_static_seq_a[i] = rgb_static[task_a[i] : task_a[i] + H * 10 : 10]
        rgb_static_seq_b[i] = rgb_static[task_b[i] : task_b[i] + H * 10 : 10]
        robot_obs_seq_a[i] = robot_obs[task_a[i] : task_a[i] + H * 10 : 10][:,:6]
        robot_obs_seq_b[i] = robot_obs[task_b[i] : task_b[i] + H * 10 : 10][:,:6]

        for k in range(robot_obs_seq_a.shape[1]):
            for l in range(robot_obs_seq_b.shape[1]):
                diff_sum = (np.abs(robot_obs_seq_a[i,k] -  robot_obs_seq_b[i,l])).sum()
                
                if diff_sum < 1.2:
                   similarity.append((i,k,l)) 
                
            

    return


def VariationalAutoEncoder(rgb_static, rgb_gripper, actions, robot_obs):

    vae = custom_VAE(32, enc_type= "resnet18")
    vae = custom_VAE.load_from_checkpoint('/home/ibrahimm/Documents/dl_lab/calvin/sg_st_weights/version_4/checkpoints/epoch=14999-step=1680000.ckpt') #sg_st_ste_actions_weights

    H = 15

    rgb_static_tensor, actions_tensor, robot_obs_tensor   = random_sampler(rgb_static, actions, robot_obs, H)

    rgb_static_first_obs = rgb_static_tensor[:,0].float()
    rgb_static_last_obs = rgb_static_tensor[:,-1].float()

    dataset = CustomDataset(rgb_static_last_obs, rgb_static_first_obs, rgb_static_tensor, actions_tensor)

    train_dataloader = DataLoader(dataset, batch_size = 15, shuffle= True, num_workers = 2)

    sg, st, _ , actions = next(iter(train_dataloader))

    vae.eval()
    #sg_reconstructed = vae(sg) # for sg_weights
    sg_reconstructed, zg = vae(sg, st) # for sg_st_weights

    return sg, sg_reconstructed, zg

if __name__ == "__main__":
    path = './gti_demos/'
    rgb_static, rgb_gripper, actions, robot_obs = read_data(path)
    
    rgb_static_tensor_resized = resize(rgb_static)

    # cv2.imshow('rgb_static', rgb_static[0])
    # cv2.waitKey(0)

    #intersection(rgb_static_tensor_resized, robot_obs)

    sg, sg_reconstructed, zg = VariationalAutoEncoder(rgb_static_tensor_resized, rgb_gripper, actions, robot_obs)

    fig, axes = plt.subplots(2, 10, figsize=(10, 2))
    axes[0][0].set_ylabel('Real', fontsize=12)
    axes[1][0].set_ylabel('Generated', fontsize=12)

    for i in range(10):
  
        ax_real = axes[0][i]
        ax_real.imshow((sg[i].type(torch.uint8)).permute(1,2,0))
        ax_real.get_xaxis().set_visible(False)
        ax_real.get_yaxis().set_visible(False)

        ax_gen = axes[1][i]
        ax_gen.imshow((sg_reconstructed[i].type(torch.uint8)).permute(1,2,0))
        ax_gen.get_xaxis().set_visible(False)
        ax_gen.get_yaxis().set_visible(False)

    plt.show()

    