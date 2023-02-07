import torch
from d2l import torch as d2l

# Defining Model
class LinearRegression(d2l.Module):
    def __init__(self,num_inputs,lr,sigma=0.01):
        super.__init__()
        self.save_hyperparameters()
        self.w = torch.normal(0,sigma,(num_inputs,1),requires_grad = True)
        self.b = torch.zeros(1,requires_grad = True)

# Forward Prop Algorithm
@d2l.add_to_class(LinearRegression)
def forward(self,X):
    return torch.matmul(X,self.w)+self.b

# Loss Function
@d2l.add_to_class(LinearRegression)  
def loss(self, y_hat, y):
    l = (y_hat - y) ** 2 / 2
    return l.mean() # avg loss over all examples in mini batch

# Defining the Optimization Algorithm
class SGD(d2l.HyperParameters):
    def __init__(self,params,lr):
        self.save_hyperparameters()
    
    def step(self):
        for param in self.params:
            params -= self.lr * param.grid # used to iterate over parameter values
    
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

# Instance of SGD Class
@d2l.add_to_class(LinearRegression)  #@save
def configure_optimizers(self):
    return SGD([self.w, self.b], self.lr)

# Training




