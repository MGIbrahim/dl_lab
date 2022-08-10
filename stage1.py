import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from torchvision.models import resnet18 as _resnet18
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


import torch
sys.path.append('../')

from VAE_train import *

H = 15
batch_size = 32
epochs = 15001
#hidden_size = 256
hidden_size = 1024
lr = 1e-3
path = 'gti_demos/'
input_size = 2 + 512
tensorboard_dir = 'rnn_model/'

class GRU(nn.Module):
    def __init__(self, n_inputs, n_neurons, n_real_outputs):
        super(GRU, self).__init__()
        
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.n_real_outputs = n_real_outputs

        self.basic_rnn = nn.GRU(self.n_inputs, self.n_neurons)
        self.FC = nn.Linear(self.n_neurons, 256)
        self.FC2 = nn.Linear(256, self.n_real_outputs)
       
        
    def forward(self, X,batch_size):  
             
        hidden = torch.zeros(1, batch_size, hidden_size).cuda()
        output, hidden = self.basic_rnn(X, hidden) 
        output = self.FC(output)
        output = self.FC2(output)
        
        return output

    
    def predict(self, X,hidden):  
        
        output, self.hidden = self.basic_rnn(X, hidden) 
        output = self.FC(output)
        
        return output, self.hidden
            
        
def main():
    
    vae = custom_VAE(32, enc_type= "resnet18")
    vae = custom_VAE.load_from_checkpoint('sg_st_weights/version_4/checkpoints/epoch=14999-step=1680000.ckpt').cuda() #sg_st_weights

    rgb_static, rgb_gripper, rel_actions, _ = read_data(path)
    batch_rgb_static_tensor_resized = resize(rgb_static)

    resnet18 = _resnet18(pretrained=True).cuda()
    resnet18 = nn.Sequential(*(list(resnet18.children())[:-2]))

    rnn = GRU(input_size,hidden_size,7).cuda()
    # rnn = GRU(17,hidden_size,7).cuda()
    
    batch_rgb_static_tensor, batch_rel_actions_tensor = random_sampler_all(batch_rgb_static_tensor_resized, rgb_gripper, rel_actions, H)
    batch_rgb_static_first_obs = batch_rgb_static_tensor[:,0].float()
    batch_rgb_static_last_obs = batch_rgb_static_tensor[:,-1].float()

    dataset = CustomDataset(batch_rgb_static_last_obs, batch_rgb_static_first_obs, batch_rgb_static_tensor, batch_rel_actions_tensor)

    train_dataloader = DataLoader(dataset, batch_size = 32, num_workers = 2)


    criterion = nn.MSELoss()
    # optimizer = optim.Adam([{'params': rnn.parameters()},{'params': resnet18.parameters()}], lr=lr)
    optimizer = optim.Adam(rnn.parameters(), lr=lr)

    writer = SummaryWriter()
      
    for e in range(epochs):

        for step, (sg, st, static , rel_actions) in enumerate(train_dataloader):
            sg, st, static , rel_actions = sg.cuda(), st.cuda(), static.cuda() , rel_actions.cuda()
            if rel_actions.size(dim=0) != 32:
                break
            rel_actions = torch.transpose(rel_actions, 0, 1)
            rel_actions = rel_actions.float()
            static = torch.transpose(static, 0, 1)
            static = static.float()
            # r_obs = torch.transpose(r_obs, 0, 1)
            # r_obs = r_obs.float()
            with torch.no_grad():
                _, latent_space = vae(sg, st)
                
            static = torch.flatten(static,0,1)
            #with torch.no_grad():
            features = resnet18(static.cuda())
            features = torch.flatten(features,1,3)
            features = features.reshape([H,batch_size,512])
            latent_space = latent_space.unsqueeze(0).repeat(H, 1, 1)
            features = torch.cat((features, latent_space),2)
            # rnn_input = torch.cat((r_obs,latent_space),2)
            
            optimizer.zero_grad()        
            output = rnn(features.cuda(), features.size(dim=1))
            # output = rnn(rnn_input.cuda(), rnn_input.size(dim=1))
                    
            loss = criterion(rel_actions.cuda(),output.cuda())
            loss.backward()
            optimizer.step()
            
            if (step%10 == 0):
                writer.add_scalar("train_loss", loss.item())  
                print("epoch",e,'Step:',step,'|', 'Loss:',loss)
                    
        if (e%50==0):   
            torch.save({
                'epoch': e,
                'model_state_dict': rnn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, f"rnn_model/rel_lr={lr}hidden={hidden_size}h={H}epoch={e}.ckpt")

        if (e%50==0):   
            torch.save({
                'epoch': e,
                'model_state_dict': resnet18.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, f"rnn_model/rel_r18lr={lr}hidden={hidden_size}h={H}epoch={e}.ckpt")
    writer.close()
if __name__ == "__main__":
    main()        
    
