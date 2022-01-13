### Rethinking the limiting dynamics of SGD: modified loss, phase space oscillations, and anomalous diffusion

<https://arxiv.org/abs/2107.09133>

Derive a continuous-time model for SGD with finite learning rates and batch sizes as an underdamped Langevin equation. Show that the key ingredient driving these dynamics is not the origianl training loss, but rather the combination of a modified loss.

## Imitiating Deep Learning Dynamics via Locally Elastic Stochastic Differential Equations

<https://arxiv.org/abs/2110.05960>

Model the evolution of features during deep learning training using a set of SDEs that each corresponds to a training sample, where each SDE constains a drift term that reflects the impact of backpropagation at an input on the features of all samples. This uncovers a sharp phase transition phenomenon regarding the intra-class impact: if the SDEs are locally elastic - the impact is more significant on samples from the same class as the input - the featrues of the training data become linearly separable, vanishing trainin loss; otherwise, the features are not separable. Also show the emergence of a simple geometric structure called the neural collapse of the features.

### Towards Theoretically Understanding Why SGD Generalizes Better Than ADAM in Deep Learning

<https://arxiv.org/abs/2010.05627>

Observe the heavy tails of gradient noise, and analyze SGD and ADAM through their Levy-deriven SDEs. Establish the escaping time of SDEs from a local basin, and show that the escaping time depends on the Radon measure of the basin positively and the heaviness of gradient noise negatively, for same basin, SGD enjoys smaller escaping time, because of geometric adaption in ADAM and exponential graident average in ADAM.

### On Large Batch Training and Sharp Minima: A Fokker-Planck Perspective

<https://arxiv.org/abs/2112.00987>

Approximate the mini-batch SGD and the momentum SGD with SDE, and use the theory of Fokker-Planck equations to develop new results on the escaping phenomenon and the relationship with large batch and sharp minima. Find that the stochastic process solution tends to converge to flatter minima regardless of the batch size in the asymptotic regime, but the convergence rate depend on the batch size.

### Scaling Properties of Deep Residual Networks

<https://arxiv.org/abs/2105.12245>

Depending some certain features of neural network architectures, like smoothness of activation function, the scaling regime is different, either neural ODE limit, SDE, or neither of these.