import torch.nn
import sys
sys.path.append('../')

#from network import Network

class RNN:
    
    def __init__(self,input_size, hidden_size, output_size):
        self.network = Network(input_size, hidden_size, output_size)
        self.criterion = torch.nn.MSELoss()
        
    def update(self,h,input,actions):
        hidden = self.network.initHidden()

        # rnn.zero_grad()

        for i in range(h):
            output, hidden = self.network(input, hidden)

        loss = self.criterion(output,actions)
        loss.backward()

        return output