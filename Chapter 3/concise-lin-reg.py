import numpy as np
from torch import nn
from d2l import torh as d2l

# using predefined layers for computing linear regression loss

# Defining the model

# LazyLinear and Linear in PyTorch allows us to construct the model easily without worrying about the amthematical details.

# The single layer Lin Reg model is fully connected. Each of its inputs is connected to each output.

class LinearRegression(d2l.Module):
    def __init__(self,lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.fill_(0)

# Define Forward Prop
@d2l.add_to_class(LinearRegression)
def forward(self,X):
    return self.net(X)

# Define Loss Function
@d2l.add_to_class(LinearRegression)
def loss(self,y,y_hat):
    fn = nn.MSELoss()
    return fn(y,y_hat)

# Define Optimizers
@d2l.add_to_class(LinearRegression)  #@save
def configure_optimizers(self):
    return nn.optim.SGD(self.parameters(), self.lr)

# Training Loop
model = LinearRegression(lr=0.3)
data = d2l.SyntheticRegressionData(w=nn.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs = 1)  
trainer.fit(model,data)

# Below, we compare the model parameters learned by training on finite data and the actual parameters that generated our dataset. To access parameters, we access the weights and bias of the layer that we need. As in our implementation from scratch, note that our estimated parameters are close to their true counterparts.
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self):
    return (self.net.weight.data, self.net.bias.data)
w, b = model.get_w_b()

print(f'error in estimating w: {data.w - w.reshape(data.w.shape)}')
print(f'error in estimating b: {data.b - b}')



