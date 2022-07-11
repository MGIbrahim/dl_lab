from torchvision.models import resnet18 as _resnet18
import torch
import torch.nn as nn
import sys
sys.path.append('../')

from rnn.RNN import RNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(st, latent_space, hidden_size, output_size, h, actions):
    
    resnet18 = _resnet18(pretrained=True)
    resnet18 = nn.Sequential(*(list(resnet18.children())[:-2]))
    
    resnet18_output = resnet18(st)
    rnn_input = torch.cat((latent_space, resnet18_output), 1)
    rnn = RNN(len(rnn_input), hidden_size, output_size )
    output = rnn.update(h, rnn_input, actions)
    
    return output
    
    
    
    
    

