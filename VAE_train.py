#from pl_bolts.datamodules import CIFAR10DataModule
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


class CustomDataset(Dataset):
    def __init__(self, batch_rgb_static_last_obs, batch_rgb_static_first_obs, actions):
        self.batch_rgb_static_last_obs = batch_rgb_static_last_obs
        self.batch_rgb_static_first_obs = batch_rgb_static_first_obs
        self.actions = actions
    
    def __len__(self):
        return self.batch_rgb_static_last_obs.shape[0]
        
    def __getitem__(self, index):
        return self.batch_rgb_static_last_obs[index], self.batch_rgb_static_first_obs[index]#, self.actions[index]

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
    
    return np.array(rgb_static), np.array(rgb_gripper), np.array(actions),\
           np.array(rel_actions), np.array(robot_obs), np.array(scene_obs)
 
def random_sampler(rgb_static, rgb_gripper, actions, robot_obs, H):
    # H is the sample length
    #indices = [141,3716], len = 3576
    # range of indices starts from 141 to 3716 - (H * 10) as steps of 10 are taken
    # and to prevent getting into index out of bounds error 
    indices = list(range(len(rgb_static) - H * 10))

    random_indices = random.choices(indices, k = len(rgb_static))

    rgb_static_tensor = torch.zeros((len(rgb_static), H, 3, rgb_static.shape[2], rgb_static.shape[3]), dtype=torch.uint8)
    #rgb_gripper_tensor = torch.zeros((len(rgb_static), H, 3, rgb_gripper.shape[2], rgb_gripper.shape[2]), dtype=torch.uint8)
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

    batch_rgb_static_tensor_resized = torch.zeros((len(rgb_static), 3, 32, 32), dtype=torch.uint8)
    
    transform = T.Compose([
        T.Resize(size = (32, 32)),
        T.PILToTensor()
    ])

    for i in range(len(rgb_static)):

        img = Image.fromarray(rgb_static[i][:,:,::-1])

        batch_rgb_static_tensor_resized[i] = transform(img)

    return batch_rgb_static_tensor_resized

def VariationalAutoEncoder(rgb_static, rgb_gripper, actions, robot_obs):

    vae = custom_VAE(32, enc_type= "resnet18")

    H = 15

    rgb_static_tensor, actions_tensor, robot_obs_tensor   = random_sampler(rgb_static, rgb_gripper, actions, robot_obs, H)

    batch_rgb_static_first_obs = rgb_static_tensor[:,0].float()
    batch_rgb_static_last_obs = rgb_static_tensor[:,-1].float()

    dataset = CustomDataset(batch_rgb_static_last_obs, batch_rgb_static_first_obs, actions_tensor)

    train_dataloader = DataLoader(dataset, batch_size = 32, shuffle= True, num_workers = 2)
    dataiter = iter(train_dataloader)
    data = dataiter.next()
    batch_rgb_static_last_obs, batch_rgb_static_first_obs = data

    print(batch_rgb_static_last_obs, batch_rgb_static_first_obs)

    trainer = Trainer()
    trainer.fit(vae, train_dataloader)


    print(vae)

if __name__ == "__main__":
    path = './gti_demos/'
    rgb_static, rgb_gripper, actions, rel_actions, robot_obs, scene_obs = read_data(path)
    
    rgb_static_tensor_resized = resize(rgb_static)

    VariationalAutoEncoder(rgb_static_tensor_resized, rgb_gripper, actions, robot_obs)


    