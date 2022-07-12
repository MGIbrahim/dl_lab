from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.models.autoencoders.basic_vae.basic_vae_module import VAE
from pytorch_lightning import Trainer
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import random



class CustomDataset(Dataset):
    def __init__(self, rgb_static, rgb_gripper, actions):
        self.rgb_static = rgb_static
        self.rgb_gripper = rgb_gripper
        self.actions = actions
    
    def __len__(self):
        assert(
            self.rgb_static.shape[0] == self.rgb_gripper.shape[0]  
        )
        assert(
            self.rgb_static.shape[0] == self.actions.shape[0]
        )
        return self.rgb_static.shape[0]
        
    def __getitem__(self, index):
        return torch.from_numpy(self.rgb_static[index]), torch.from_numpy(self.rgb_gripper[index]), torch.from_numpy(self.actions[index])

#Loading GTI demonstrations
def read_data(path):
    indices = [141,3716] #filtered gti_demos
    indices = list(range(indices[0], indices[1] + 1))
    data = ['rgb_static', 'rgb_gripper']

    idx = indices[0]
    i = 0
    len_indices = indices[-1] - indices[0]
    rgb_static = [0] * (len_indices+1)
    rgb_gripper = [0] * (len_indices+1)
    actions = [0] * (len_indices+1)
    for idx in indices:
        t = np.load(f'{path}/episode_{idx:07d}.npz', allow_pickle=True)
        print(f"episode_{indices[i]:07d}.npz")
        for d in data:
            if d == 'rgb_static':
                rgb_static[i]  = t[d][:,:,::-1]
            elif d == 'rgb_gripper':
                rgb_gripper[i]  = t[d][:,:,::-1]

        actions[i]  = t['actions']
        i+=1
        
    # Get Images as features and actions as targets
    # rgb_static and rgb_gripper being X_train data
    # actions being the y_train
    return np.array(rgb_static).astype('float32'), np.array(rgb_gripper).astype('float32'), np.array(actions).astype('float32')

def random_sampler(rgb_static, rgb_gripper, actions, batch_size, H):
    # H is the sample length
    #indices = [141,3716]
    indices = list(range(len(rgb_static) - H))
    batch_indices = random.sample(indices, batch_size)
    #rgb_static_H = []
    #rgb_gripper_H = []
    #actions_H = []
    #batch_rgb_static_tensor = torch.zeros((batch_size, H, rgb_static.shape[1], rgb_static.shape[2]))
    batch_rgb_static_tensor = torch.zeros((batch_size, H, rgb_static.shape[1], rgb_static.shape[2], 3))
    #batch_rgb_gripper_tensor = torch.zeros((batch_size, H, rgb_gripper.shape[1], rgb_gripper.shape[2]))
    batch_rgb_gripper_tensor = torch.zeros((batch_size, H, rgb_gripper.shape[1], rgb_gripper.shape[2], 3))
    #batch_actions_tensor = torch.zeros((batch_size, H, actions.shape[1]))
    batch_actions_tensor = torch.zeros((batch_size, H, actions.shape[1]))
    
    i = 0

    for index in batch_indices:
        rgb_static_H = []
        rgb_static_H.extend(rgb_static[index:index + H])
        rgb_static_H = np.array(rgb_static_H).astype('float32')
        batch_rgb_static_tensor[i] = torch.from_numpy(rgb_static_H)
        
        rgb_gripper_H = []
        rgb_gripper_H.extend(rgb_gripper[index:index + H])
        rgb_gripper_H = np.array(rgb_gripper_H).astype('float32')
        batch_rgb_gripper_tensor[i] = torch.from_numpy(rgb_gripper_H)

        actions_H = []
        actions_H.extend(actions[index:index + H])
        actions_H = np.array(actions_H).astype('float32')     
        batch_actions_tensor[i] = torch.from_numpy(actions_H)

        i+=1

    #return rgb_static_H, rgb_gripper_H, actions_H
    return batch_rgb_static_tensor#, batch_rgb_gripper_tensor, batch_actions_tensor

def rgb2gray(rgb):
    """ 
    this method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721])
    return gray.astype('float32') 

def sample(mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

def VariationalAutoEncoder(rgb_static, rgb_gripper, actions):
    
    #dataset = CustomDataset(rgb_static, rgb_gripper, actions)
    #dataloader = DataLoader(dataset, batch_size = 1)
    #dataiter = iter(dataloader)
    #data = dataiter.next()
    #rgb_static, rgb_gripper, actions = data
    #print(rgb_static, rgb_gripper, actions)
    
    #goal_state_rgb_static_H = rgb_static[-1]
    #goal_state_rgb_gripper_H = rgb_gripper[-1]
    #goal_state_actions_H = actions[-1]

    vae = VAE(200, enc_type= "resnet18")
    vae = vae.from_pretrained("cifar10-resnet18")

    #Training Loop:
    n_iter = 300
    H = 15
    batch_size = 32
    #for epoch in range(n_iter):
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

    #print(fw_pass_first)
    print(z_first)

    #print(fw_pass_last)
    print(z_last)
    
    
    
    
    #vae = VAE(32, lr=0.00001)
    
    #vae = vae.from_pretrained("cifar10-resnet18")
    #

    
    #Training Loop:
    #num_epochs = 2
    #total

    ##dm = CIFAR10DataModule(".")
    #dm.prepare_data()
    #dm.setup("fit")
    #dataloader = dm.train_dataloader()
    #
    #X, _ = next(iter(dataloader))
    #vae.eval()
    #X_hat = vae(X)
    #
    #fig, axes = plt.subplots(2, 10, figsize=(10, 2))
    #axes[0][0].set_ylabel('Real', fontsize=12)
    #axes[1][0].set_ylabel('Generated', fontsize=12)
    #
    #for i in range(10):
    #  
    #  ax_real = axes[0][i]
    #  ax_real.imshow(np.transpose(X[i], (1, 2, 0)))
    #  ax_real.get_xaxis().set_visible(False)
    #  ax_real.get_yaxis().set_visible(False)
    #
    #  ax_gen = axes[1][i]
    #  ax_gen.imshow(np.transpose(X_hat[i].detach().numpy(), (1, 2, 0)))
    #  ax_gen.get_xaxis().set_visible(False)
    #  ax_gen.get_yaxis().set_visible(False)
    #plt.show()


if __name__ == "__main__":
    path = '/home/ibrahimm/Documents/dl_lab/calvin/gti_demos/'
    rgb_static, rgb_gripper, actions = read_data(path)
    #rgb_static_gray = rgb2gray(rgb_static)
    #rgb_gripper_gray = rgb2gray(rgb_gripper)
    



    #VariationalAutoEncoder(rgb_static_gray, rgb_gripper_gray, actions)
    VariationalAutoEncoder(rgb_static, rgb_gripper, actions)


    