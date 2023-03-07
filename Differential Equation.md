### Rethinking the limiting dynamics of SGD: modified loss, phase space oscillations, and anomalous diffusion

<https://www.arxiv.org/abs/2107.09133>

Derive a continuous-time model for SGD with finite learning rates and batch sizes as an underdamped Langevin equation. Show that the key ingredient driving these dynamics is not the origianl training loss, but rather the combination of a modified loss.

## Imitiating Deep Learning Dynamics via Locally Elastic Stochastic Differential Equations

<https://www.arxiv.org/abs/2110.05960>

Model the evolution of features during deep learning training using a set of SDEs that each corresponds to a training sample, where each SDE constains a drift term that reflects the impact of backpropagation at an input on the features of all samples. This uncovers a sharp phase transition phenomenon regarding the intra-class impact: if the SDEs are locally elastic - the impact is more significant on samples from the same class as the input - the featrues of the training data become linearly separable, vanishing trainin loss; otherwise, the features are not separable. Also show the emergence of a simple geometric structure called the neural collapse of the features.

### Towards Theoretically Understanding Why SGD Generalizes Better Than ADAM in Deep Learning

<https://www.arxiv.org/abs/2010.05627>

Observe the heavy tails of gradient noise, and analyze SGD and ADAM through their Levy-deriven SDEs. Establish the escaping time of SDEs from a local basin, and show that the escaping time depends on the Radon measure of the basin positively and the heaviness of gradient noise negatively, for same basin, SGD enjoys smaller escaping time, because of geometric adaption in ADAM and exponential graident average in ADAM.

### On Large Batch Training and Sharp Minima: A Fokker-Planck Perspective

<https://www.arxiv.org/abs/2112.00987>

Approximate the mini-batch SGD and the momentum SGD with SDE, and use the theory of Fokker-Planck equations to develop new results on the escaping phenomenon and the relationship with large batch and sharp minima. Find that the stochastic process solution tends to converge to flatter minima regardless of the batch size in the asymptotic regime, but the convergence rate depend on the batch size.

### Scaling Properties of Deep Residual Networks

<https://www.arxiv.org/abs/2105.12245>

Depending some certain features of neural network architectures, like smoothness of activation function, the scaling regime is different, either neural ODE limit, SDE, or neither of these.

### What Happens after SGD Reaches Zero Loss? --A Mathematical Framework

<https://www.arxiv.org/abs/2110.06914>

Gives a general framework for implicit bias analysis, allowing a complete characterization for the regularization effect of SGD around convergent manifold using a SDE describing the limiting dynamics of the parameters.

### Infinitely Deep Bayesian Neural Network with Stochastic Differential Equations

<https://www.arxiv.org/abs/2102.06559>

In approximate inference of continuous depth Bayesian neural networks, the uncertainty of each weights follow a SDE. Demonstrate gradient-based SVI in infinite parameter setting, produces arbitrarily flexible approximate posteriors, and derive gradient estimator approaching zero variance as approximate posterior approaches the true posterior.

### Framing RNN as a kernel method: A neural ODE approach

<https://www.arxiv.org/abs/2106.01202>

Show that under appropriate conditions, the solution of a RNN can be viewed as a linear function of a specific feature set of the input sequence, known as the signature, framing RNN as a kernel method in a suitable RKHS. Obtain theoretical guarantees on generalization and stability.

### Augmented Neural ODE

<https://proceedings.neurips.cc/paper/2019/file/21be9a4bd4f81549a9d1d241981cec3c-Paper.pdf>

Show that Neural ODEs learn representation preserving the topology of the input space, hence there is function that Neural ODE can not represent. Solve this limitation by Augmented Neural ODE.

### Quantized convolutional neural networks through the lens of partial differential equations

<https://www.arxiv.org/abs/2109.00095>

Explore ways to improved quantized CNNs using PDE-based perspective, harnessing the total variation approach to apply edge-aware smoothing.

### Do Residual Neural Networks discretize Neural Ordinary Differential Equations?

<https://www.arxiv.org/abs/2205.14612>

Quantify the distance between the solution of Neural ODE and ResNet's hidden state. This tight bound shows that it does not go to 0 with large depth if the residual functions are not smooth with depth, but show that this smoothness is preserved by gradient descent for a ResNet with linear residual functions and small enough initial loss. Consider the use of memory-free discrete discrete adjoint method to train a ResNet, and show that this method succeeds ar large depth, and Heun's method allows for better gradient estimation with the adjoint method.

### High-dimensional limit theorems for SGD: Effective dynamics and critical scaling

<https://www.arxiv.org/abs/2206.04030>

Prove the limit theorems for the trajectories of summary statistics of SGD as the dimension goes to infinity, which ields both ballistic (ODE) and diffusive (SDE) limits, depending on the initialization and step size. Find a critical scaling regime for the step size, below gives ballistic and above gives diffusive diagrams. Demonstrate on classification via two-layer NN.

### Scaling ResNets in the Large-depth Regime

<https://www.arxiv.org/abs/2206.06929>

Show that with standard initialization, the only nontrivial dynamics is given by aL = 1/sqrt(L), where other choices lead to exploration or identity mapping. This corresponds to continuous-time limit to a neural stochastic differential equation.

### Do Residual Neural Networks discretize Neural Ordinary Differential Equations?

<https://www.arxiv.org/abs/2205.14612>

Quantify the distance between ResNet's hidden state trajectory and the solution of its corresponding Neural ODE, which is tight but does not converges to zero with depth N if the residual function are not smooth with depth. But show that this smoothness is preserved by gradient descent with linear residual functions and small enough initial loss, showing the implicit regularization towards limit Neural ODE.

### Vanilla feedforward neural networks as a discretization of dynamics systems

<https://www.arxiv.org/abs/2209.10909>

Show that the not only ResNet, but the classical network structure can also be a numerical discretization of dynamic systems, which is based the properties of the leaky-ReLU function.

### Surprising Instabilities in Training Deep Networks and a Theoretical Analysis

<https://www.arxiv.org/abs/2206.02001>

To analyze the numerical error due to the floating point computation, derive the gradient descent PDE of the NN learning, and analyze it using the Pon-Neumann analysis to study its stability.

### Asymptotic Analysis of Deep Residual Networks

<https://www.arxiv.org/abs/2212.08199>

Show the existence of scaling regime for trained weights different from those implicitly assumed. Study the convergence, that one can obtain either ODE, SDE, or neither of these which is existence of diffusive regime.

### Convergence Analysis for Training Stochastic Neural Networks via Stochastic Gradient Descent

<https://www.arxiv.org/abs/2212.08924>

Consider the training of discretization of SDE as stochastic neural network via sample-wise back propagation, with adjoint backward SDE. Derive the convergence analysis with and without the convexity analysis, and show that the SNN training steps should be proportional to the squares of the number of layers.

### An SDE for Modeling SAM: Theory and Insights

<https://www.arxiv.org/abs/2301.08203>

Derive the continuous-time model for SAM and unnormalized variant USAM, for both full-batch and mini-batch. These models offer explanation wh SAM prefers flat minima over sharp ones, that they minize an implicitly regularized loss with a Hessian-dependent noise structure. 

### From high-dimensional & mean-field dynamics to dimensionless ODEs: A unifying approach to SGD in two-layers networks

<https://www.arxiv.org/abs/2302.05882>

Analyze the SGD dynamics of a two-layer network trained on Gaussian data, via the sufficient statistics for the population risk. This bridges gradient flow to high dimensional regime, and overparameterized case and the intermediate regimes between these. The infinite-width dynamics remain close to the low-dimensional subspace spanned by the targe principal directions.

### Toward Equation of Motion for Deep Neural Networks: Continuous-time Gradient Descent and Discretization Error Analysis

<https://www.arxiv.org/abs/2210.15898>

Derive a counter term that cancels the discretization error between GF and GD for DNNs, and obtain the equation of motion that precisely describes the discrete learning dynamics. Also derive the discretization error.