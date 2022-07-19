import sys
sys.path.append('../')

from policy.rnn.RNN import RNN
from VAE_test import *
from torch import nn
from torchvision.models import resnet18 as _resnet18
import matplotlib.pyplot as plt
from torch import optim

batch_size=32
H=4
epochs =200

rgb_static, rgb_gripper, actions = read_data('gti_demos/')
batch_rgb_static_tensor, batch_rgb_gripper_tensor, batch_actions_tensor = random_sampler(rgb_static=rgb_static, rgb_gripper=rgb_gripper, 
                                                                                         actions=actions, batch_size=batch_size, H=H)
batch_actions_tensor = torch.transpose(batch_actions_tensor,0 , 1)

batch_rgb_static_tensor = torch.transpose(batch_rgb_static_tensor,0 , 1)
batch_rgb_static_tensor = torch.transpose(batch_rgb_static_tensor,2 , 4)
batch_rgb_static_tensor = torch.transpose(batch_rgb_static_tensor,3 , 4)

criterion = nn.MSELoss()

resnet18 = _resnet18(pretrained=True)
resnet18 = nn.Sequential(*(list(resnet18.children())[:-2]))

features = resnet18.forward(batch_rgb_static_tensor[0])

rnn = RNN(25088,256,7).cuda()
optimizer = optim.Adam(rnn.parameters())
for e in range(epochs):
    hidden = rnn.initHidden()
    hidden = hidden.detach()
    output_tensor = torch.zeros((H,batch_size, 7))
    for i in range(H):
        optimizer.zero_grad()
        features = resnet18.forward(batch_rgb_static_tensor[i])
        features = torch.flatten(features,1,3)
        output, hidden = rnn.forward(features.cuda(),hidden.cuda())
        output_tensor[i] = output

    loss = criterion(batch_actions_tensor,output_tensor)
    loss.backward()
    optimizer.step()
    if (e%10 == 0):
        print('epoch:',e,'|', 'Loss:',loss)
        print(batch_actions_tensor,output_tensor)




