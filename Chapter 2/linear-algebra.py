# Scalars
import torch
x = torch.tensor(3.0)
y = torch.tensor(2.0)
x + y, x * y, x / y, x**y

# Matrices
A = torch.arange(6).reshape(3, 2)

# Matrices with same transpose and original s called a symmetric matrix

# Tensors
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
B = A.clone()  # Assign a copy of `A` to `B` by allocating new memory

# The elementwise product of two matrices is called their Hadamard product (denoted). Below, we spell out the entries of the Hadamard product of two matrices 

A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
B = A.clone()  # Assign a copy of `A` to `B` by allocating new memory

print(A*B)

# Reduction
x = torch.arange(3, dtype=torch.float32)
print(x.sum())

# Sum along rows
print(A.sum(axis=0))

# Sum along columns
print(A.sum(axis=1))

# Summing along rows and columns is same as summing over all elements
A.sum(axis=[0, 1]) == A.sum() # Same as `A.sum()`

# Non reduction sum - keeping the dimensions same
sum_A = A.sum(axis=1, keepdims=True)

# Dot Products
y = torch.ones(3, dtype = torch.float32)
torch.sum(x*y)

# matrix vector solution
print(torch.mv(A,x))

# matrix matrix multiplication
B=torch.ones((3,4))
torch.mm(A,B)

# norms
u = torch.tensor([3.0,-4.0])
torch.norm(u) #l2 norm
torch.abs(u).sum() #l1 norm
torch.norm(torch.ones(4,9)) # Frobenius norm





