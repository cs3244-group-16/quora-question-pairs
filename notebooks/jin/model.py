
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SAlayer(nn.Module):
    def __init__(self,...):
        super(SAlayer, self).__init__()
        # in & out features
        # alpha for loss function
        # self.concat # True for all layers except output layer
        # initialisation of weights

        # loss function
        pass
    
    def forward(self, x):
        # attention
        # with softmax
        pass

class transformer(nn.Module):
    def __init__(self, ...):
        super(transformer, self).__init__()
        # no. of attention heads
        # nfeat?
        # number of hidden layers
        # number of classes -- 2
        # alpha # for layer
        pass

    
    def forward(self, x):
        # layers
        # return loss value
        pass


