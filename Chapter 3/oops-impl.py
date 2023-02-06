# Inspired by open-source libraries such as PyTorch Lightning, on a high level we wish to have three classes: (i) Module contains models, losses, and optimization methods; (ii) DataModule provides data loaders for training and validation; (iii) both classes are combined using the Trainer class, which allows us to train models on a variety of hardware platforms. Most code in this book adapts Module and DataModule. We will touch upon the Trainer class only when we discuss GPUs, CPUs, parallel training, and optimization algorithms.

import time
import numpy as np
import torch as nn
from d2l import torch as d2l

# Register the functions as methods of class
def add_to_class(Class):
    def wrapper(obj):
        setattr(Class,obj.__name__,obj)
    return wrapper

# How to use this 
class A:
    def __init__(self):
        self.b = 1

a = A()

# create an instance
@add_to_class(A)
def do(self):
    print('Class attribute "b" is', self.b)
a.do()

# Second utility

# The second one is a utility class that saves all arguments in a class’s __init__ method as class attributes. This allows us to extend constructor call signatures implicitly without additional code.
class HyperParameters:
    def save_parameters(self,ignore=[]):
        raise NotImplemented

class B(d2l.HyperParameters):
    def __init__(self, a, b, c):
        self.save_hyperparameters(ignore=['c'])
        print('self.a =', self.a, 'self.b =', self.b)
        print('There is no self.c =', not hasattr(self, 'c'))

b = B(a=1, b=2, c=3)

# Plot progress with ProgressBoard
class ProgressBoard(d2l.HyperParameters):
    def __init__(self,xlabel = None,ylabel = None,xlim = None,ylim = None,xscale = "linear",yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()
    def draw(self,x,y,label,every_n = 1):
        raise NotImplemented

board = d2l.ProgressBoard('x')
for x in np.arange(0, 10, 0.1):
    board.draw(x, np.sin(x), 'sin', every_n=2)
    board.draw(x, np.cos(x), 'cos', every_n=10)

class Module(nn.Module,d2l.HyperParameters):
    def __init__(self,plot_train_epoch=2,plot_valid_per_epoch = 1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()
    
    def loss(self,y_hat,y):
        return NotImplementedError
    
    def forward(self,y_hat,y):
        assert hasattr(self,'net'),'Neural Network Defined'
        return self.net(X)
    
        def plot(self, key, value, train):
                assert hasattr(self, 'trainer'), 'Trainer is not inited'
                self.board.xlabel = 'epoch'
                if train:
                    x = self.trainer.train_batch_idx / \
                        self.trainer.num_train_batches
                    n = self.trainer.num_train_batches / \
                        self.plot_train_per_epoch
                else:
                    x = self.trainer.epoch + 1
                    n = self.trainer.num_val_batches / \
                        self.plot_valid_per_epoch
                self.board.draw(x, value.to(d2l.cpu()).detach().numpy(),
                                ('train_' if train else 'val_') + key,
                                every_n=int(n))
    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        raise NotImplementedError

# Data

class DataModule(d2l.HyperParameters):
        def __init__(self, root='../data', num_workers=4):
            self.save_hyperparameters()

        def get_dataloader(self, train):
            raise NotImplementedError
        
        def train_dataloader(self):
            return self.get_dataloader(train=True)

        def val_dataloader(self):
            return self.get_dataloader(train=False)
# Trainer
class Trainer(d2l.HyperParameters):  #@save
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def fit_epoch(self):
        raise NotImplementedError

