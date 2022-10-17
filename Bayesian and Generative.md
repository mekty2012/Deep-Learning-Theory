### Global inducing point varaitional posteriors for Bayesian neural networks and deep Gaussian processes

<https://www.arxiv.org/abs/2005.08140>

Consider the optimal approximate posterior over the top-layer weights, and show that it exhibits strong dependencies on the lower-layer weights, and also on Deep GP.
This idea uses learned global inducing points that is propagated through layer. 

### Exact priors of finite neural networks

<https://www.arxiv.org/abs/2104.11734>

Derive exact solutions for the output priors for individual input examples of a class of fintie fully-connected feedforward Bayesian neural networks.

### Bayesian Neural Network Priors Revisited

<https://arxiv.org/abs/2102.06571>

Find that CNN weights display strong spatial correlations, while FCNNs display heavy-tailed weight distributions.

### Pathologies in priors and inference for Bayesian transformers

<https://arxiv.org/abs/2110.04020>

Weight-space inference in transformers does not work well, regardless of the approximate posterior. Also find that the prior is at least partially at fault but that it is very hard to find well-specified weight priors for these models.

### A global convergence theory for deep ReLU implicit networks via over-parameterization

<https://arxiv.org/abs/2110.05645>

Show that a randomly initialized gradient descent converges to a global minimum at a linear rate for the square loss function if the implicit neural network is over parameterized.

### Non-convergence of stochastic gradient descent in the training of deep neural networks

<https://arxiv.org/abs/2006.07075>

Show that stochastic gradient descent can fail if depth is much larger than their width, and the number of random initialization does not increase to infinity fast enough.

### Uniform convergence may be unable to explain generalization in deep learning

<https://arxiv.org/abs/1902.04742>

Present examples of overparameterized linear classifiers and neural networks trained by gradient descent where uniform convergence provably cannot explain generalization, even if we take into account the implicit bias of GD. 

### Asymptotics of representation learning in finite Bayesian neural networks

<https://arxiv.org/abs/2106.00651>

Argue that the leading finite-width corrections to the average feature kernels for any Bayesian network with linear readout and Gaussian likelihood have a largely universal form. Illustrate this explicitly for linear MLP or CNN and single nonlinear hidden layer.

### Bayesian Optimization of Function Networks

<https://www.arxiv.org/abs/2112.15311>

Design new bayesian optimization which models the nodes using Gaussian process, in the setting that network takes significant time to evaluate. Show that the posterior is efficiently maximized via sample average approximation, and that method is asymptotically consistent.

### Wide Mean-Field Bayesian Neural Networks Ignore the Data

<https://www.arxiv.org/abs/2202.11670>

Show that mean-field variational inference fails when network width is large and the activation is odd. In specific, with odd activation and homoscedastic Gaussian likelihood, show that optimal mean-field variational posterior predictive distribution converges to the prior predictive distribution as the width tends to infinity.

### Pre-training helps Bayesian optimization too

<https://www.arxiv.org/abs/2109.08215>

For selecting a prior for Bayesian optimization, consider the senario with data from similar functions to pre-train a tighter distribution a priori. Show a bounded regret of bayesian optimization with pre-trained priors. 

### Depth induces scale-averaging in overparameterized linear Bayesian neural networks

<https://www.arxiv.org/abs/2111.11954>

Interpret finite deep linear Bayesian neural networks as data-dependent scale mixtures of Gaussian process predictors across output channels, and study the representation learning, connecting the limit results in previous studies.

### Exact marginal prior distributions of finite Bayesian neural networks

<https://www.arxiv.org/abs/2104.11734>

Derive the exact solutions for the function space priors for finite fully-connected Bayesian neural networks. Linear activation gives simple expression in terms of the Meijer G-function, and the prior of a finite ReLU network is a mixture of the priors of linear networks of smaller widths.

### How Tempering Fixes Data Augmentation in Bayesian Neural Networks

<https://www.arxiv.org/abs/2205.13900>

Identify two factors influencing the strength of the cold posterior effect, the correlated nature of augmentations and the degree of invariance of the employed model to such transformations. Analyzing simplified setting, prove that tempering implicitly reduces the misspecification arising from modeling augmentations as i.i.d. data, which mimics the role of the effect sample size. 

### Asymptotic Properties for Bayesian Neural Network in Besov Space

<https://arxiv.org/abs/2206.00241>

Show that the BNN using spike-and-slab prior has consistency with nearly minimax cnovergence rate when the true function is in the Besov space.

### Wide Bayesian neural networks have a simple weight posterior: theory and accelerated sampling

<https://arxiv.org/abs/2206.07673>

Introduce repriorisation, which transform the BNN posterior to a distribution whose KL divergence to the BNN prior vanishes as layer widths grow. This analytic simplicity complements the NNGP behaviour, and using this repriorisation, implement MCMC posterior sampling algorithm which mixes faster the wider the BNN is, and is effective in high dimensions.

### Analysis of Discriminator in RKHS Function Space for Kullback-Leibler Divergence Estimation

<https://www.arxiv.org/abs/2002.11187>

Use GAN to estimate KL divergence, argue that high fluctuations in the estimates are a consequence of not controlling the complexity of the discriminator function space. Provide a theoretical underpinning and remedy for this problem by constructing a discriminator in the RKHS.

### Forward Super-Resolution: How Can GANs Learn Hierarchical Generative Models for Real-World Distributions

<https://arxiv.org/abs/2106.02619>

Prove that when a distribution has a structure that referred as Forward Super-Resolution, then training GANs using gradient descent ascent can indeed learn this distribution efficiently both in terms of sample and time complexities.

### On the Convergence of Gradient Descent in GANs: MMD GAN As a Gradient Flow

<https://arxiv.org/abs/2011.02402>

Show that parametric kernelized gradient flow provides a descent direction minimizing the MMD on a statistical manifold of probability distributions.

### On some theoretical limitations of Generative Adversarial Networks

<https://arxiv.org/abs/2110.10915>

Provide a new result based on Extreme Value Theory showing that GANs can't generate heavy tailed distributions.

### Diagnosing and Fixing Manifold Overfitting in Deep Generative Models

<https://arxiv.org/abs/2204.07172>

Investigate the pathology of maximum likelihood training with dimensionality mismatch, and prove that degenerate optima are achieved where the manifold itself is learned, but not the distribution on it. Propose a two-step procedure of dimensionality reduction and maximum-likelihood density estimation, and prove that this procedure recover the data-generating distribution in the nonparameteric regime.

### An error analysis of generative adversarial networks for learning distributions

<https://www.arxiv.org/abs/2105.13010>

Establish the convergence rate of GANs under collection of integral probability metrics defined through Hoelder class lke Wasserstein distance. Also show that GANs are able to adaptively learn data distributions with low-dimensional structures or have Hoelder densities, with proper architecture. In particular, show that for low-dimensional structure, the convergence rate depends on intrinsic dimension, not ambient dimension.

### Why GANs are overkill for NLP

<https://www.arxiv.org/abs/2205.09838>

Show that, while it seems that maximizing likelihood is different than minimizing distinguishability criteria, this distinction is artifical and only holds for limited models. And show that minimizing KL-divergence is a more efficient approach to effectively minimizing the same distinguishability.

### The Gaussian equivalence of generative models for learning with shallow neural networks

<https://www.arxiv.org/abs/2006.14709>

Establish rigorous conditions for the Gaussian Equivalence between single layer neural network and Gaussian models, with convergence rate. Use this equivalence to derive a closed set of equations of generalization performance of two-layer neural network trained with SGD or full batch pre-learned feature. 

### Score-Based Generative Models Detect Manifolds

<https://www.arxiv.org/abs/2206.01018>

Show that the SDEs arising from the diffusion models can exactly approximate the score, under the assumption on the SDE ensuring the reversal process, and the uniqueness of the reverse SDE.

### High-Dimensional Distribution Generation Through Deep Neural Networks

<https://www.arxiv.org/abs/2107.12466>

Show that every d-dimensional probability distribution of bounded support can be generated through deep ReLU networks. The proof is based on generalization of space-filling approach.

### Dynamics of Fourier Modes in Torus Generative Adversarial Networks

<https://www.arxiv.org/abs/2209.01842>

Decompose the objective function of adversary min-max game defining a periodic GAN into its Fourier series, and study the dynamics of truncated Fourier series. Using this, approximate the real flow and identify the main features of convergence of the GAN, showing that convergence orbits in GANs are small perturbations of periodic orbits so the Nash equilibria are spiral attractors.

### On the Nash equilibrium of moment-matching GANs for stationary Gaussian processes

<https://arxiv.org/abs/2203.07136>

Study the Nash equilibrium where discriminator defined on the second-order statistical moments. Show that they can result non-existence of Nash equilibrium, or existence of consistent non-Nash equilibrium, or existence and uniqueness of consistent Nash equilibrium. The symmetry property of the generator family determines which of the results hold.

### Mitigating the Effects of Non-Identiability on Inference for Bayesian Neural Networks with Latent Variables

<https://www.arxiv.org/abs/1911.00569>

Demonstrate that in the limit of infinite data, the posterior mode over the NN weights and latent variables is asymptotically biased away from the ground-truth. Due to this bias, the traditional inferences may yield parameters that generalize poorly.

### Convergence of score-based generative modeling for general data distributions

<https://arxiv.org/abs/2209.12381>

Give polynomial convergence guarantees for denoising diffusion models without functional inequalities or strong smoothness assumptions. With L2 accurate score estimates, obtain Wasserstein distance guarantee for any bounded support or fast decaying tails, and total variation guarantee with further smoothness assumptions.

### Scale-Invariant Bayesian Neural Networks with Connextivity Tangent Kernel

<https://arxiv.org/abs/2209.15208>

The generalization bound by loss landscapes usually can be chaned arbitrarily to the scale of a parameter. Propose new prior distribution that is invariant to scaling transformations, giving generalization bound that works for more practical class of transformations such as weight decay.

### A Variational Perspective on Generative Flow Networks

<https://www.arxiv.org/abs/2210.07992>

Define the variational objectives for GFNs in terms of KL distribution between the forward and backward distribution, and show that variational inference is equivalent to minimizing the trajectory balance objective when sampling trajectories from the forward model. 