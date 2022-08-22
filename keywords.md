### (Over-/Under)parameterization

Neural network usually have many parameters, sometimes more than the number of data. 
We call these overparameterization, and the converse, underparameterization.
The exact definition varies for papers, but usually we either assume that number of parameter is much larger than the number of data point, or the target solution is exactly learnable.

### Neural Collapse

In the classification problem with _deep_ neural networks, sometimes the activation values of latter layers concentrate around its mean, conditional on the class. 
This behavior is called neural collapse, and the reason or usage for this behavior is recently studied.

### Neural Network Gaussian Process

Under the infinite width limit, with the random initialization of parameters, the network's distribution converges to Gaussian process.
This behavior has been used to understand the Bayesian neural networks, and the intuitions on neural network's power at initialization.

### Neural Tangent Kernel

Under the infinite width limit, the training dynamics of neural network simplifies to simple linear ODE, which involves matrix computed from some kernel, named _Neural Tangent Kernel_. 
This phenomenon has been well used to analyze the training dynamics of neural network.

### Generalization Bound

Since the dataset is finite where real data is a distribution, due to the error from sampling, test error will be larger than training error.
To theoretically bound these two quantities' difference, several tools including VC dimension or Rademacher complexity are developed, and are used in learning theory.