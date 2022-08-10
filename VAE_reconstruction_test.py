from pl_bolts.datamodules import CIFAR10DataModule
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
    def __init__(self, batch_rgb_static_last_obs, batch_rgb_static_first_obs, batch_rgb_static_tensor, actions):
        self.batch_rgb_static_last_obs = batch_rgb_static_last_obs
        self.batch_rgb_static_first_obs = batch_rgb_static_first_obs
        self.batch_rgb_static_tensor = batch_rgb_static_tensor
        self.actions = actions
    
    def __len__(self):
        return self.batch_rgb_static_last_obs.shape[0]
        
    def __getitem__(self, index):
        return self.batch_rgb_static_last_obs[index], self.batch_rgb_static_first_obs[index], self.batch_rgb_static_tensor[index], self.actions[index]  

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
                rgb_static[i]  = t[d][:,:,::-1] # Converts from BGR to RGB
            elif d == 'rgb_gripper':
                rgb_gripper[i]  = t[d][:,:,::-1] # Converts from BGR to RGB

        actions[i]  = t['actions']
        i+=1
        
    return np.array(rgb_static), np.array(rgb_gripper), np.array(actions)

def random_sampler(rgb_static, rgb_gripper, actions, batch_size, H):
    
    # H is the sample length
    indices = list(range(len(rgb_static) - H))

    random_indices = random.choices(indices, k = len(rgb_static) * 3) # Create a list of random indices with length of rgb_static * 3 = 3576 * 3

    batch_rgb_static_tensor = torch.zeros((len(rgb_static) * 3, H, 3, rgb_static.shape[2], rgb_static.shape[3]), dtype=torch.uint8)
    #batch_rgb_gripper_tensor = torch.zeros((batch_size, H, 3, rgb_gripper.shape[2], rgb_gripper.shape[3]), dtype=torch.uint8)
    batch_actions_tensor = torch.zeros((len(rgb_static) * 3, H, actions.shape[1]))
    
    i = 0
    for index in random_indices:
        batch_rgb_static_tensor[i] = rgb_static[index:index + H]
        batch_actions_tensor[i] = torch.from_numpy(actions[index:index + H])
        i+=1

    return batch_rgb_static_tensor, batch_actions_tensor 

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


def VariationalAutoEncoder(rgb_static, rgb_gripper, actions):

    vae = custom_VAE(32, enc_type= "resnet18")
    #vae = custom_VAE.load_from_checkpoint('/home/ibrahimm/Documents/dl_lab/calvin/sg_weights/epoch=5563-step=1869503.ckpt') #sg_weights
    vae = custom_VAE.load_from_checkpoint('/home/ibrahimm/Documents/dl_lab/calvin/sg_st_step_actions_weights/epoch=10368-step=394022.ckpt') #sg_st_weights


    H = 15
    batch_size = 32

    batch_rgb_static_tensor, batch_actions_tensor = random_sampler(rgb_static, rgb_gripper, actions, batch_size, H)

    batch_rgb_static_first_obs = batch_rgb_static_tensor[:,0].float()
    batch_rgb_static_last_obs = batch_rgb_static_tensor[:,-1].float()

    #print(batch_rgb_static_last_obs, batch_rgb_static_first_obs)

    dataset = CustomDataset(batch_rgb_static_last_obs, batch_rgb_static_first_obs, batch_rgb_static_tensor, batch_actions_tensor)

    train_dataloader = DataLoader(dataset, batch_size = 1, num_workers = 2)

    #for step, (sg, st, _ , actions) in enumerate(train_dataloader):
        #print("in loop")

    sg, st, _ , actions = next(iter(train_dataloader))

    vae.eval()
    #sg_reconstructed = vae(sg) # for sg_weights
    sg_reconstructed, zg = vae(sg, st) # for sg_st_weights

    return sg, sg_reconstructed, zg


if __name__ == "__main__":
    path = './gti_demos/'
    rgb_static, rgb_gripper, actions = read_data(path)
    
    batch_rgb_static_tensor_resized = resize(rgb_static)


    sg, sg_reconstructed, zg  = VariationalAutoEncoder(batch_rgb_static_tensor_resized, rgb_gripper, actions)

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