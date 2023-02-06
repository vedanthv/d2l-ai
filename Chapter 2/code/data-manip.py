import torch

x = torch.arange(12,dtype = torch.float32) # tensors are meant for CPU based computation

print(x.numel()) # prints the number of elements in the tensor

print(x.shape())

X = x.reshape((3,4))

torch.zeros((2,3,4))

torch.ones((1,2,3))

# We often wish to sample each element randomly (and independently) from a given probability distribution. For example, the parameters of neural networks are often initialized randomly. The following snippet creates a tensor with elements drawn from a standard Gaussian (normal) distribution with mean 0 and standard deviation 1.

torch.randn(3,4)

torch.exp(x)

# Axis = 0 [concat along rows] and Axis = 1 concat along columns
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)

# Saving Space and In Place Operations
# Fortunately, performing in-place operations is easy. We can assign the result of an operation to a previously allocated array Y by using slice notation: Y[:] = <expression>. To illustrate this concept, we overwrite the values of tensor Z, after initializing it, using zeros_like, to have the same shape as Y.

Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))

# both the ids in this case is same. its an inplace operation

A = X.numpy()
B = torch.from_numpy(A)
type(A), type(B)

# First one is nd array, second is tensor