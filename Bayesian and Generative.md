## Bayesian

### Global inducing point varaitional posteriors for Bayesian neural networks and deep Gaussian processes

<https://www.arxiv.org/abs/2005.08140>

Consider the optimal approximate posterior over the top-layer weights, and show that it exhibits strong dependencies on the lower-layer weights, and also on Deep GP.
This idea uses learned global inducing points that is propagated through layer. 

### Exact priors of finite neural networks

<https://www.arxiv.org/abs/2104.11734>

Derive exact solutions for the output priors for individual input examples of a class of fintie fully-connected feedforward Bayesian neural networks.

### Bayesian Neural Network Priors Revisited

<https://www.arxiv.org/abs/2102.06571>

Find that CNN weights display strong spatial correlations, while FCNNs display heavy-tailed weight distributions.

### Pathologies in priors and inference for Bayesian transformers

<https://www.arxiv.org/abs/2110.04020>

Weight-space inference in transformers does not work well, regardless of the approximate posterior. Also find that the prior is at least partially at fault but that it is very hard to find well-specified weight priors for these models.

### Asymptotics of representation learning in finite Bayesian neural networks

<https://www.arxiv.org/abs/2106.00651>

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

<https://www.arxiv.org/abs/2206.00241>

Show that the BNN using spike-and-slab prior has consistency with nearly minimax cnovergence rate when the true function is in the Besov space.

### Wide Bayesian neural networks have a simple weight posterior: theory and accelerated sampling

<https://www.arxiv.org/abs/2206.07673>

Introduce repriorisation, which transform the BNN posterior to a distribution whose KL divergence to the BNN prior vanishes as layer widths grow. This analytic simplicity complements the NNGP behaviour, and using this repriorisation, implement MCMC posterior sampling algorithm which mixes faster the wider the BNN is, and is effective in high dimensions.

### Mitigating the Effects of Non-Identiability on Inference for Bayesian Neural Networks with Latent Variables

<https://www.arxiv.org/abs/1911.00569>

Demonstrate that in the limit of infinite data, the posterior mode over the NN weights and latent variables is asymptotically biased away from the ground-truth. Due to this bias, the traditional inferences may yield parameters that generalize poorly.

### Scale-Invariant Bayesian Neural Networks with Connextivity Tangent Kernel

<https://www.arxiv.org/abs/2209.15208>

The generalization bound by loss landscapes usually can be chaned arbitrarily to the scale of a parameter. Propose new prior distribution that is invariant to scaling transformations, giving generalization bound that works for more practical class of transformations such as weight decay.

### Do Bayesian Neural Networks Need To Be Fully Stochastic?

<https://www.arxiv.org/abs/2211.06291>

Prove that the partially stochastic neural networks with only n stochastic biases are universal probabilistic predictors for n-dimensional predictive problems.

### Bayesian Interpolation with Deep Linear Networks

<https://www.arxiv.org/abs/2212.14457>

Consider the Bayeisan inference using deep linear network. For the zero noise, show that both the predictive posterior and the ayesian model evidence can be written in closed form in terms of Meijer-G functions. This result is non-asymptotic, and holds for all depth, width, giving exact solution to Bayesian interpolation using a deep GP with a Euclidean covariance at each layer. 

### Free energy of Bayesian Convolutional Neural Network with Skip Connection

<https://www.arxiv.org/abs/2307.01417>

Derive the Bayesian free energy of CNN with and without skip connection, and show that upper bound of free energy does not depend on the overparameterization and the generalization error has similar property.

### Understanding Pathologies of Deep Heteroskedastic Regression

<https://www.arxiv.org/abs/2306.16717>

Show that the instabilities of over-parameterized heteroskedastic regression that either mean or variance collapses, are already exists in a field theory of an overparameterized conditional Gaussian model. Derive a free energy that can be solved numerically, and shows existence of phase transition of behaviors of regressors on the regularization strenghts, which emphasizes the necessity of careful regression.

### Bayes-optimal Learning of Deep random Networks of Extensive-width

<https://www.arxiv.org/abs/2302.00375>

Derive closed-form expression for the Bayes-optimal test error in the asymptotic limit where input dimension, number of sample, and width are proportionally large, ad show that optimally regularized ridge regression and kernel regression achieve Bayes-optimal performance, while logistic loss gives near-optimal test error for classification.

### Masked Bayesian Neural Networks : Theoretical Guarantee and its Posterior Inference

<https://www.arxiv.org/abs/2305.14765>

Design new node-sparse BNN models whose posterior concentration rate is near minimax optimal and adaptive to the smoothness of the true model.

## Generative Models

### Analysis of Discriminator in RKHS Function Space for Kullback-Leibler Divergence Estimation

<https://www.arxiv.org/abs/2002.11187>

Use GAN to estimate KL divergence, argue that high fluctuations in the estimates are a consequence of not controlling the complexity of the discriminator function space. Provide a theoretical underpinning and remedy for this problem by constructing a discriminator in the RKHS.

### Forward Super-Resolution: How Can GANs Learn Hierarchical Generative Models for Real-World Distributions

<https://www.arxiv.org/abs/2106.02619>

Prove that when a distribution has a structure that referred as Forward Super-Resolution, then training GANs using gradient descent ascent can indeed learn this distribution efficiently both in terms of sample and time complexities.

### On the Convergence of Gradient Descent in GANs: MMD GAN As a Gradient Flow

<https://www.arxiv.org/abs/2011.02402>

Show that parametric kernelized gradient flow provides a descent direction minimizing the MMD on a statistical manifold of probability distributions.

### On some theoretical limitations of Generative Adversarial Networks

<https://www.arxiv.org/abs/2110.10915>

Provide a new result based on Extreme Value Theory showing that GANs can't generate heavy tailed distributions.

### Diagnosing and Fixing Manifold Overfitting in Deep Generative Models

<https://www.arxiv.org/abs/2204.07172>

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

<https://www.arxiv.org/abs/2203.07136>

Study the Nash equilibrium where discriminator defined on the second-order statistical moments. Show that they can result non-existence of Nash equilibrium, or existence of consistent non-Nash equilibrium, or existence and uniqueness of consistent Nash equilibrium. The symmetry property of the generator family determines which of the results hold.

### Convergence of score-based generative modeling for general data distributions

<https://www.arxiv.org/abs/2209.12381>

Give polynomial convergence guarantees for denoising diffusion models without functional inequalities or strong smoothness assumptions. With L2 accurate score estimates, obtain Wasserstein distance guarantee for any bounded support or fast decaying tails, and total variation guarantee with further smoothness assumptions.

### A Variational Perspective on Generative Flow Networks

<https://www.arxiv.org/abs/2210.07992>

Define the variational objectives for GFNs in terms of KL distribution between the forward and backward distribution, and show that variational inference is equivalent to minimizing the trajectory balance objective when sampling trajectories from the forward model. 

### How Well Generative Adversarial Networks Learn Distributions

<https://www.arxiv.org/abs/1811.03179>

Nonparametrically, derive the optimal minimax rates for distribution estimation under the adversarial framework. Parametrically, estabilsh a theory for general neural network classes that characterized the interplay on the choice of generator and discriminator pair.

### Certifiably Robust Variational Autoencoders

<https://www.arxiv.org/abs/2102.07559>

Derive actionable bounds on the minimal size of an input perturbation required to change a VAE's reconstruction by more than an allowed amount. Then show how these parameters can be controlled, providing a mechanism to ensure desired level of robustness.

### alpha-GAN: Convergence and Estimation Guarantees

<https://www.arxiv.org/abs/2205.06393>

Prove a two-way correspondence between general CPE loss function GANs and the minimization of associated f-divergence. Show that the Arimoto divergences induced by a alpha-GAN equivalently converge for all alpha, and provide estimation bounds.

### Distribution Approximation and Statistical Estimation Guarantees of Generative Adversarial Networks

<https://www.arxiv.org/abs/2002.03938>

Consider the approximation of data distributions that have densities in Hoelder space, show that assuming both discriminator and generator are properly chosen, GAN becomes the consistent estimator of data distribution under strong discrepancy metrics including Wasserstein-1 distance. Moreover when data distribution exhibits low-dimensional structure, show that GANs are capable to capture this strcture and achieve a fast statistical convergence, free of curse of the ambient dimensionality.

### Convergence of denoising diffusion models under the manifold hypothesis

<https://www.arxiv.org/abs/2208.05314>

The theoretical analysis of denoising diffusion models assume that the target density is absolutely continuous w.r.t. Lebesgue measure, which does not cover setting when the target distribution is supported on a lower-dimensional manifold or is empirical distribution. Provide the first convergence result for more general setting, which is quantitative bounds on the Wasserstein distance of order one between target data distribution and the generative diftribution.

### identifiability of deep generative models without auxiliary information

<https://www.arxiv.org/abs/2206.10044>

Show that for a generative models with universal approximation capabilities, the side information is not necessary. Prove the identifiability of the entire generative model without side information, only data.

### On PAC-Bayesian reconstruction guarantees for VAEs

<https://www.arxiv.org/abs/2202.11455>

Analyze the VAE's reconstruction ability for unseen test data with PAC-Bayes theory. Provide generalisation bounds on the theoretical reconstruction error and provide insights on the regularisation effect of VAE objective.

### Embrace the Gap: VAEs Perform Independent Mechanism Analysis

<https://www.arxiv.org/abs/2206.02416>

Prove that in the limit of near-determinstic decoders, optimal encoder approximately inverts the decoder. Using this phenomenon, show that ELBO converges to a regularized log-likelihood, and allow VAE to perform independent mechanism analysis that adds an inductive bias towards column-orthogonal Jacobians for decoders.

### Approximation bounds for norm constrained neural networks with applications to regression and GANs

<https://www.arxiv.org/abs/2201.09418>

Prove approximation capacity of ReLU NN with norm constraint on the weights, especially upper and lower bound of approximation error of smooth function class, where lower bound comes from Rademacher complexity. Using this bounds, analyze convergence of regression and distribution estimation by GANs.

### Learning (Very) Simple Generative Models Is Hard

<https://www.arxiv.org/abs/2205.16003>

Show that under the SQ model, no polynomial time algorithm can solve the generative model problem, even when the hidden layer of true function is only logarithmically many. Show this by stacking the discrete distribution that matches small moments to N(0, I).

### Rate of convergence for density estimation with generative adversarial networks

<https://www.arxiv.org/abs/2102.00199>

Prove a sharp oracle inequality of Jensen-Shannon inequality between true density and vanilla GAN estimate. Also study the rate of convergence, where the JS-divergence decays as fast by the term determined by number of samples, and smoothness of true density.

### Optimal precisions for GANs

<https://www.arxiv.org/abs/2207.10541>

Using the geometric measure theory, prove a sufficient condition for optimality where the dimension of the latent space is larger than the number of modes.

### Geometry of Score Based Generative Models

<https://www.arxiv.org/abs/2302.04411>

Show that the forward and backward process of score based generative model is Wasserstein gradient flow. 

### PAC-Bayesian generalization Bounds for Adversarial Generative Models

<https://www.arxiv.org/abs/2302.08942>

Extend the PAC-Bayeisan theory to generative models based on Wasserstein distance and the total variation distance. These apply to WGAN and Energy based GANs.

### On Deep Generative Models for Approximation and Estimation of Distributions on Manifolds

<https://www.arxiv.org/abs/2302.13183>

Assuming that data are supported on a low-dimensional manifold, prove statistical guarantee of generative networks under Wasserstein-1loss, and the convergence rate is given by intrinsic dimension.

### Mathematical analysis of singualrities in the diffusion model under the submanifold assumption

<https://www.arxiv.org/abs/2301.07882>

Show that the analytic mean drift function in DDPM and the score function in SGM asymptotically blow up in the final stages, for singular distributions that are concentrated in low dimensional manifold. Derive new target and associated loss, which remains bounded for such singular distributions.

### Approximating Probability Distributions by using Wasserstein Generative Adversarial Networks

<https://www.arxiv.org/abs/2103.10060>

Establish quantified generalization bound for Wasserstein distance between generated and target distributions. WGANs has higher requirement for capacity of discriminators than generators, and overly deep and wide generators may be worse than low-capacity generators if discriminators are not strong enough.

### Towards Faster Non-Asymptotic Convergence for Diffusion-Based Generative Models

<https://www.arxiv.org/abs/2306.09251>

For deterministic sampler, show the convergence rate proportional to 1/T, and for stochastic models with 1/sqrt(T), with minimal assumption on the target data. Also design accelerated variant where deterministic sampler has 1/T^2 convergence and stochastic sampler has 1/T.

### Local Convergence of Gradient Descent-Ascent for Training Generative Adversarial Networks

<https://www.arxiv.org/abs/2305.08277>

Study the local dynamics of GDA for kernel-based discriminator and GAN, and finds effect of learnin rate, regularization, bandwidth of neural network, on the local convergence rate.