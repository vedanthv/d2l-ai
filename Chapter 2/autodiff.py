# Creating the function
import torch
x = torch.arange(4,requires_grad = True)
print(x.grad) # prints none

y = 2*torch.dot(x,x)

y.backward()
print(x.grad)

print(x.grad == 4*x)

x.grad.zero_()

y = x.sum()
y.backward()
print(x.grad)

x.grad.zero_()
y = x * x
y.backward(gradient=torch.ones(len(y)))
print(x.grad)

# Detaching Computation

# sometimes we may need to calculate derivative of x wrt z but we dont want y as an intermidiate variable for computation of x wrt y. So in this case we detach computation

x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
print(x.grad == u)

# gradients and control flow

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size = (),requires_grad = True)
d = f(a)
d.backward()

print(a.grad==(d/a))





