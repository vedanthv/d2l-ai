## Weight Decay

Now that we have characterized the problem of overfitting, we can introduce our first regularization technique. Recall that we can always mitigate overfitting by collecting more training data. 

However, that can be costly, time consuming, or entirely out of our control, making it impossible in the short run. For now, we can assume that we already have as much high-quality data as our resources permit and focus the tools at our disposal even when the dataset is taken as a given.

**It operates by restricting the values that a parameter can take**

We can measure the complexity of functions by the distance of its parameters from 0 if we initialize the values of the function f by 0.

Measuring the complexity of a linear function is often done by its norm value.

We can add the l2 norm as a penalty term to ensure that we minimize the loss.

So we can replace our original decision of prediction loss on training labels by minimizing the sum of the prediction loss and the penlaty term.

If our weight vector grows very large, then our learning algorithm focuses on reducing the penalty term and not the weight vector which is good.

```math
L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.
```

How does the model trade off the standard loss for the penalty term? We use the regularization constant.

```math
L(\mathbf{w}, b) + \frac{\lambda}{2} \|\mathbf{w}\|^2.
```

For lambda = 0 we get the same function back without any penalty term. For lambda > 0 we restrict the size of norm of w.

L2 Regularization is called classic **Ridge Regression Model** and for L1 Regulaization its called classic **Lasso Regression Model**

L1 penalizes some weights all the way to 0 so it can be used for feature selection.

**Weight Decay After Regularization**

```math
\begin{aligned}
\mathbf{w} & \leftarrow \left(1- \eta\lambda \right) \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right).
\end{aligned}
```

Complexity of a function can be measured from f = 0 with default parameters.

Functional Analysis deals with how much precision should be ensured when measuring from f = 0

The main reason for using L2 Norm is that it penalizes the features that are irrelavant to a larger extent.

This helps us with feature selection functionality.

**Mini Batch Stochastic Gradient**

```math
\begin{aligned}
\mathbf{w} & \leftarrow \left(1- \eta\lambda \right) \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right).
\end{aligned}
```
We try to reduce w value to 0 and hence its called weight decay.

This biases our learning algorithm towards models that distribute weight evenly across a larger number of features. In practice, this might make them more robust to measurement error in a single variable.

Our optimization algorithm decreases the weight constantly until it reaches 0 or its minima.

Sometimes an additional b^2 penalty term is also included.

Although regularization may not be equivalent to weight decay for other optimization algorithms, the idea of regularization through shrinking the size of weights still holds true.

In this chapter we discussed only linear functions, but for Reproducing Kernel Hilbert Spaces, they would not scale very well.
