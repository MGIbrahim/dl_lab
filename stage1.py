import sys
from pl_bolts.datamodules import CIFAR10DataModule
#from pl_bolts.models.autoencoders.basic_vae.basic_vae_module import VAE
from pl_bolts.models.autoencoders import VAE
from pytorch_lightning import Trainer
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import random

import torch
sys.path.append('../')

from gti_dataset.gti_dataset import DemosDataset
from VAE_test import *


#from VAE import VAE
#vae = VAE()

vae = VAE(200, enc_type= "resnet18")
vae = vae.from_pretrained("cifar10-resnet18")

epochs = 1000
batch_size = 32
H = 15
face_dataset = DemosDataset(root_dir='gti_demos/')
rgb_static, rgb_gripper, actions = read_data('gti_demos/')

for i in range(epochs):
    #batch_rgb_static_tensor, batch_rgb_gripper_tensor, batch_actions_tensor = random_sampler(rgb_static=rgb_static, rgb_gripper=rgb_gripper, 
                                                                                         #actions=actions, batch_size=batch_size, H=15)
    #batch_rgb_static_tensor_state = torch.tensor(np.zeros((batch_size, 200, 200, 3))).cuda()
    #batch_rgb_static_tensor_goal = torch.tensor(np.zeros((batch_size, 200, 200, 3))).cuda()
    #for i in range(batch_size):
        #batch_rgb_static_tensor_state[i,:,:,:]= batch_rgb_static_tensor[i,0,:,:,:]
        #batch_rgb_static_tensor_goal[i,:,:,:]= batch_rgb_static_tensor[i,14,:,:,:]
    #batch_rgb_static_tensor_state = torch.transpose(batch_rgb_static_tensor_state, 1, 3)
    #batch_rgb_static_tensor_goal = torch.transpose(batch_rgb_static_tensor_goal, 1, 3)

    #elbo = vae.training_step(batch_rgb_static_tensor_state, batch_rgb_static_tensor_goal)
    #print(elbo)

    batch_rgb_static_tensor = random_sampler(rgb_static, rgb_gripper, actions, batch_size, H)
    batch_rgb_static_first_obs = torch.movedim(batch_rgb_static_tensor[:,0], 3, 1) # torch.Size([15, 3, 200, 200])
    batch_rgb_static_last_obs = torch.movedim(batch_rgb_static_tensor[:,-1], 3, 1) # torch.Size([15, 3, 200, 200])
    
    fw_pass_first, x_first, z_first = vae.forward(batch_rgb_static_first_obs) # First Observation State of size batch_size
    fw_pass_last, x_last, z_last = vae.forward(batch_rgb_static_last_obs) # Last Observation State of size batch_size

    x = torch.zeros((batch_size, x_first.shape[1]*2))

    for i in range(x_first.shape[0]):
        x[i] = torch.cat((x_first[i], x_last[i]))
    
    enc_out_dim: int =  1024  # Original Number 512
    latent_dim: int = 512   # Original Number 256

    fc_mu = nn.Linear(enc_out_dim, latent_dim)
    fc_var = nn.Linear(enc_out_dim, latent_dim)

    mu = fc_mu(x)
    log_var = fc_var(x)
    p, q, z = sample(mu, log_var) 
    print(z_first)
    print(z_last) 
        
# x=face_dataset.__getitem__(12)
# tensor_x = torch.Tensor(x)
