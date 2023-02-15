import torch
from torch import nn
from torch.nn import nn
from torch.nn import functional as F
from d2l import torch as d2l

class SoftmaxRegression(d2l.Classfier):
    def __init__(self,num_outputs,lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(),nn.LazyLinear(num_outputs))

    def forward(self,X):
        return self.net(X)
    
# Loss function
@d2l.add_to_class(d2l.Classifier)  #@save
def loss(self, Y_hat, Y, averaged=True):
    Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
    Y = Y.reshape((-1,))
    return F.cross_entropy(
        Y_hat, Y, reduction='mean' if averaged else 'none')


