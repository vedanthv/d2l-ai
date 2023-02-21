## Chapter 5 - *Multilayer Perceptrons*

### Hidden Layers

In softmax regression we map inputs via single affine transformations, add a bias and perform softmax operation.

But in softmax regression, linearity is a big assumption.

### Limitations of Linear Models

- Sometimes Linear Models Make sense.Sometimes that makes sense. For example, if we were trying to predict whether an individual will repay a loan, we might reasonably assume that all other things being equal, an applicant with a higher income would always be more likely to repay than one with a lower income.

- This situation looks linear for sure but it sure isnt monotonically linear.
An increase in income from $0 to $50,000 is certainly more than increase in income from 1 million to 1.05 million.

- And yet despite the apparent absurdity of linearity here, as compared with our previous examples, it is less obvious that we could address the problem with a simple preprocessing fix. 

- That is, because the significance of any pixel depends in complex ways on its context (the values of the surrounding pixels)

### Activation Functions

- Te activation function decides if we activate a neuron or not during the calculation of the weighted sum.

Some popular activation functions are : 

1. ReLU

```math
\operatorname{ReLU}(x) = \max(x, 0).
```

Informally, the ReLU function retains only positive elements and discards all negative elements by setting the corresponding activations to 0. To gain some intuition, we can plot the function. As you can see, the activation function is piecewise linear.

<img src = "http://d2l.ai/_images/output_mlp_76f463_18_0.svg">

When the input is negative, the derivative of the ReLU function is 0, and when the input is positive, the derivative of the ReLU function is 1. 

Note that the ReLU function is not differentiable when the input takes value precisely equal to 0. 

In these cases, we default to the left-hand-side derivative and say that the derivative is 0 when the input is 0.

<img src = "http://d2l.ai/_images/output_mlp_76f463_33_0.svg">

One good thing about ReLU is that the function eiither lets the entire input pass through or does not let it pass through.

2. Sigmoid Function

The sigmoid function transforms its inputs, for which values lie in the domain 
, to outputs that lie on the interval (0, 1). For that reason, the sigmoid is often called a squashing function: it squashes any input in the range (-inf, inf) to some value in the range (0, 1):

```math
\operatorname{sigmoid}(x) = \frac{1}{1 + \exp(-x)}.
```

- Sigmoids are mostly used in binary classification to find out the binary probability values.

- However, the sigmoid has mostly been replaced by the simpler and more easily trainable ReLU for most use in hidden layers. 

- Much of this has to do with the fact that the sigmoid poses challenges for optimization (LeCun et al., 1998) since its gradient vanishes for large positive and negative arguments.

<img src = "http://d2l.ai/_images/output_mlp_76f463_48_0.svg">

**Derivative Of Sigmoid**

```math
\frac{d}{dx} \operatorname{sigmoid}(x) = \frac{\exp(-x)}{(1 + \exp(-x))^2} = \operatorname{sigmoid}(x)\left(1-\operatorname{sigmoid}(x)\right).
```

<img src = "http://d2l.ai/_images/output_mlp_76f463_63_0.svg">

### Tanh Function

Like the sigmoid function, the tanh (hyperbolic tangent) function also squashes its inputs, transforming them into elements on the interval between -1 and 1:

```math
\operatorname{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}.
```

<img src = "http://d2l.ai/_images/output_mlp_76f463_78_0.svg">

### Initializing Model Parameters

To begin, we will implement an MLP with one hidden layer and 256 hidden units. Both the number of layers and their width are adjustable (they are considered hyperparameters). Typically, we choose the layer widths to be divisible by larger powers of 2.

### Forward Propogation

Forward propagation (or forward pass) refers to the calculation and storage of intermediate variables (including outputs) for a neural network in order from the input layer to the output layer. 

We now work step-by-step through the mechanics of a neural network with one hidden layer.

Step 1 : 

```math
\mathbf{z}= \mathbf{W}^{(1)} \mathbf{x},
```

Step 2 : Adding Activation Function

```math
\mathbf{h}= \phi (\mathbf{z}).
```

Step 3 : Finding the output

```math
\mathbf{o}= \mathbf{W}^{(2)} \mathbf{h}.
```

Step 4 : Loss Function

```math
s = \frac{\lambda}{2} \left(\|\mathbf{W}^{(1)}\|_F^2 + \|\mathbf{W}^{(2)}\|_F^2\right),
```

Step 5 : Implementing Regularization

```math
J = L + s.
```

Therefore when training neural networks, after model parameters are initialized, we alternate forward propagation with backpropagation, updating model parameters using gradients given by backpropagation. 

Note that backpropagation reuses the stored intermediate values from forward propagation to avoid duplicate calculations. 

One of the consequences is that we need to retain the intermediate values until backpropagation is complete. 

This is also one of the reasons why training requires significantly more memory than plain prediction. 

Besides, the size of such intermediate values is roughly proportional to the number of network layers and the batch size. 

Thus, training deeper networks using larger batch sizes more easily leads to out of memory errors.

