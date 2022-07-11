import sys

import torch
sys.path.append('../')

from gti_dataset.gti_dataset import DemosDataset
from VAE_test import *


from VAE import VAE
vae = VAE()

epochs = 1000
batch_size = 32
face_dataset = DemosDataset(root_dir='gti_demos/')
rgb_static, rgb_gripper, actions = read_data('gti_demos/')

for i in range(epochs):
    batch_rgb_static_tensor, batch_rgb_gripper_tensor, batch_actions_tensor = random_sampler(rgb_static=rgb_static, rgb_gripper=rgb_gripper, 
                                                                                         actions=actions, batch_size=batch_size, H=15)
    batch_rgb_static_tensor_state = torch.tensor(np.zeros((batch_size, 200, 200, 3))).cuda()
    batch_rgb_static_tensor_goal = torch.tensor(np.zeros((batch_size, 200, 200, 3))).cuda()
    for i in range(batch_size):
        batch_rgb_static_tensor_state[i,:,:,:]= batch_rgb_static_tensor[i,0,:,:,:]
        batch_rgb_static_tensor_goal[i,:,:,:]= batch_rgb_static_tensor[i,14,:,:,:]
    batch_rgb_static_tensor_state = torch.transpose(batch_rgb_static_tensor_state, 1, 3)
    batch_rgb_static_tensor_goal = torch.transpose(batch_rgb_static_tensor_goal, 1, 3)

    elbo = vae.training_step(batch_rgb_static_tensor_state, batch_rgb_static_tensor_goal)
    print(elbo)
        
        
# x=face_dataset.__getitem__(12)
# tensor_x = torch.Tensor(x)
