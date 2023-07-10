### On the Stability Properties and the Optimization Landscape of Training Problems with Squared Loss for Neural Networks and General Nonlinear Conic Approximation Schemes

<https://www.arxiv.org/abs/2011.03293>

With the assumption of nonlinear conic approximation and unrealizable label vectors, show that a training problem with squared loss is necessarily unstable, i.e., its solution set depends discontinuously on the label vector in the training data. 

### Convex Geometry and Duality of Over-parameterized Neural Networks

<https://www.arxiv.org/abs/2002.11219>

Prove that an optimal solution to the regularized training problem can be characerized as extreme points of a convex set, so simple solutions are encouraged via its convex geometrical properties. 

### Spurious Local Minima are Common in Two-Layer ReLU Neural Networks

<https://www.arxiv.org/abs/1712.08968>

Show that two-layer ReLU networks w.r.t. the squared loss has local minima, even if the input distribution is standard Gaussian, dimension is arbitrarilty large, and orthonormal parameter vectors, using computer-assisted proof.

### Unveiling the structure of wide flat minima in neural networks

<https://www.arxiv.org/abs/2107.01163>

Show that wide flat minima arise as complex extensive structures from the coalscence of minima aroung 'high-margin' configurations. Despite being exponentially rare compared to zero-margin ones, high-margin minima tend to concentrate in particular regious, surrounded by other solutions of smaller margin.

### Entropic gradient descent algorithms and wide flat minima

<https://www.arxiv.org/abs/2006.07897>

Show that gaussian mixture classification's Bayes optimal pointwise estimators belongs to minimizers in wide flat regions, found by applying maximum flatness algorithms. Then using entropy-SGD and replicated-SGD, improve the generalization error.

### Escape saddle points by a simple gradient-descent based algorithm

<https://www.arxiv.org/abs/2111.14069>

Propose a simple gradient-based algorithm that outputs an epsilon approximate second-order stationary point, which is an idea of implementing a robust Hessian power method using only gradients, which can find negative curvature near saddle points.

### When Are Solutions Connected in Deep Networks?

<https://www.arxiv.org/abs/2102.09671>

Show that under generic assumptions on the features of intermediate layers, it suffices that the last two hiddne layers have order of sqrt(N) neurons, and if subsets of features at each layer are linearly separable, then no over-parameterization is needed to show the connectivity. 

### Convergence rates for the stochastic gradient descent method for non-convex opjective functions

<https://www.arxiv.org/abs/1904.01517>

Prove the local convergence to minima and estimates on the rate of convergence in the case of not necessarily globally convex nor contracting objective functions.

### Convergence proof for stochastic gradient descent in the training of deep neural networks with ReLU activation for constant target functions

<https://www.arxiv.org/abs/2112.07369>

Prove that under assumptions that the learning rates are sufficiently small but not L1 summable and target function is a constant function, the expectation of the risk converges to zero as step increases to infinity.

### On the existence of global minima and convergence analyses for gradient descent methods in the training of deep neural networks

<https://www.arxiv.org/abs/2112.09684>

Prove convergence of risk in gradient descent, when input data's probability distribution is piecewise polynomial, target function is also piecewise polynomial, and with at least one regular global minimum. Also show that there is global minimum for Lipscitz continuous function for shallow NN, and finally prove that gradient flow DE converges with polynomial rate.

### Taming neural networks with TUSLA: Non-convex learning via adaptive stochastic gradient Langevin algorithms

<https://www.arxiv.org/abs/2006.14514>

Use tamed unadjusted stocahstic Langevin algorithm to train NN, provide non-asymptotic analysis of convergence properties and finite-time guarantees for TUSLA to find approximate minimizer.

### On the Double Descent of Random Features Models Trained with SGD

<https://www.arxiv.org/abs/2110.06910>

Derive non-asymptotic error bound of random feature regression in high dimension with SGD training, show that due to unimodal variance and monotonic decrement of bias, there is double descent phenomenon.

### The Implicit Regularization of Momentum Gradient Descent with Early Stopping

<https://www.arxiv.org/abs/2201.05405>

Study the implicit regularization of momentum gradient flow, show that its tendency is closer to ridge than gradient descent. Moreover prove that under t=sqrt(2/lambda) where lambda is tuning parameter of ridge regression, the risk of MGF is no more than 1.54 times that of ridge.

### On the Global Convergence of Gradient Descent for Over-parameterized Models using Optimal Transport

<https://www.arxiv.org/abs/1805.09545>

Consider the discretization of unknown measure as mixture of particles, and a continuous time gradient descent on their weights and positions. Show that in this many-particle limit, the gradient flow converges to global minimizers involving Wasserstein gradient flow.

### On the Periodic Behavior of Neural Network Training with Batch Normalization with Weight Decay

<https://www.arxiv.org/abs/2106.15739>

Show that combined use of batch normalization and weight decay may result in periodic behavior of optimization behaviors. Derive the condition of this behavior both experimentally and theoretically.

### Stability of Deep Neural Networks via discrete rough paths

<https://www.arxiv.org/abs/2201.07506>

Based on stability bound of total p-variation of trained weights, interpret residual network as solutions to difference equations.

### Convergence of Deep Convolutional Neural Networks

<https://www.arxiv.org/abs/2109.13542>

Show that convergence of deep convolutional neural networks reduces to convergence of inifniite products of matrices with increasing sizes, and establish sufficient conditions for convergence of such infinite products.

### Improved Complexities for Stochastic Conditional Gradient Methods under Interpolation-like Conditions

<https://www.arxiv.org/abs/2006.08167>

Analyze stochastic conditional gradient method for constrained optimization problems, show that when the objective function is convex, it requires quadratic number of steps w.r.t. error.

### Power-law escape rate of SGD

<https://www.arxiv.org/abs/2105.09557>

Using the property of SGD noise to derive a SDE with simpler additive noise, show that the log loss barrier which is log ratio between local minimum loss and saddle loss, determines the escape rate of SGD from local minimum. 

### Saddle-to-Saddle Dynamics in Deep Linear Networks: Small Initialization Training, Symmetry, and Sparsity

<https://www.arxiv.org/abs/2106.15933>

Consider training dynamics of Deep Linear Networks in low variance initialization, conjecture a saddle-to-saddle dynamics, gradient descent visits the neighborhoods of a sequence of saddles each corresponding to increasing rank, and reaches sparse global minimum. This conjecture is supported by a theorem for dynamics between first two saddles.

### How many degrees of freedom do we need to train deep networks: a loss landscape perspective

<https://www.arxiv.org/abs/2107.05802>

Find that there is a phase transition of dimensionality required to train the NN, which depends on initial loss and final loss. Theoretically explain the origin with its dependency, using Gordon's escape theorem that the training dimension plus Gaussian width of desired loss set must exceed the toal number of parameters to have large success probability.

### Finite-Sum Optimization: A New Perspective for Convergence to a Global Solution

<https://www.arxiv.org/abs/2202.03524>

Using reformulation of optimization allowing for a new recursive algorithmic framework, prove convergence to epsilon global minimum with cubic time.

### Anticorrelated Noise Injection for Improved Generalization

<https://www.arxiv.org/abs/2202.02831>

Experimentally find that anticorrelated perturbation generalizes significantly better than GD and standard uncorrelated PGD, with theoretical analysis that Anti-PGD moves to wider minima where GD or PGD remains suboptimal regions.

### On Margin Maximization in Linear and ReLU Networks

<https://www.arxiv.org/abs/2110.02732>

It was shown that homogeneous networks trained with the exponential loss or the logistic loss's gradient flow converges to a KKT point of the max margin problem. This paper show that, the KKT point is not even a local optimum of max margin problem in many cases, and identify settings where local or global optimum can be guaranteed.

### Training neural networks using monotone variational inequality

<https://www.arxiv.org/abs/2202.08876>

Instead of traditional loss function, reduce training to another problem with convex structure, solving a monotone variational inequality. The solution can be founded by efficient procedure, with performance guarantee of l2 and l infty bound on model recovery accuracy and prediction accuracy with shallow linear neural networks. Also propose a practical algorithm called stochastic variational inequality, which gives competitive performance on SGD for FCNN and GNNs.

### Demystifying Batch Normalization in ReLU Networks: Equivalent Convex Optimization Models and Implicit Regularization

<https://www.arxiv.org/abs/2103.01499>

Analyze Batch Normalization through the lens of convex optimization. Introduce an analytic framework based on convex duality, obtain exact convex representation of weight-decay regularized ReLU networks with BN, trainable in polytime. Show that optimal layer weights can be obtained as simple closed form formulas in high-dimensional overparameterized regimes. 

### Connecting Optimization and Generalization via Gradient Flow Path Length

<https://www.arxiv.org/abs/2202.10670>

Propose a framework to connect optimization with generalization by analyzing the generalization error based on the length of optimization trajectory under the gradient flow algorithm after convergence. Show that with proper initialization, gradient flow converges following a short path with explicit length estimate. Such estimate induces length-based generalization bound, showing that short path after convergence are associated with good generalization.

### Local SGD Optimizes Overparameterized Neural Networks in Polynomial Time

<https://www.arxiv.org/abs/2107.10868>

Prove that Local SGD can optimize deep neural networks with ReLU activation function in polynomial time. Show that traditional approach using gradient Lipscitzness does not hold in ReLU nets, but the change between local model and average model will not change too much.

### Optimal Learning Rates of Deep Convolutional Neural Networks: Additive Ridge Functions

<https://www.arxiv.org/abs/2202.12119>

Show that for additive ridge functions, CNNs followed by one FC layer with ReLU activation can reach optimal mini-max rates.

### Benign Underfitting of Stochastic Gradient Descent

<https://www.arxiv.org/abs/2202.13361>

Prove that there exist problem instances where SGD solution exhibits both empirical risk and generalization gap of Omega(1), and show that SGD is not algorithmically stable, and its generalization ability can't be explained by uniform convergence or other known generalization bound techniques. 

### On the Power and Limitations of Random Features for Understanding Neural Networks

<https://www.arxiv.org/abs/1904.00687>

Review the techniques using random feature that the optimization dynamics behave as the initial random value, and argue that random features can't be used to learn even single neuron unless network size is exponentially large

### The loss landscape of deep linear neural networks: a second-order analysis

<https://www.arxiv.org/abs/2107.13289>

Study the optimization landscape of deep linear neural networks with the square loss. Characterize all critical points, which are global minimizers, strict saddle points, and non-strict saddle points, and all the associated critical values. 

### The Hidden Convex Optimization Landscape of Two-Layer ReLU Neural Networks: an Exact Characterization of the Optimal Solutions

<https://www.arxiv.org/abs/2006.05900>

Prove that all globally optimal two-layer ReLU neural networks can be performed by solving a convex optimization problem with cone constraint. Establish that the Clarke stationary points found by SGD correspond to global optimum of a subsampled convex problem, provide polynomial-time algorithm for checking whether network is at global minimum of training loss, provide an explicit construction of a continuous path between global minimum and any point, and characterize the minimal size of hidden layer so that loss landscape has no spurious valleys.

### Resonance in Weight Space: Covariate Shift Can Drive Divergence of SGD with Momentum

<https://www.arxiv.org/abs/2203.11992>

Show that SGDm under covariate shift with fixed step size can be unstable and diverge, in particular, that SGDm under covariate shift is a parameteric oscilator, so can suffer from resonance. Approximate the learning system as a time-varying system of ODEs with applications of existing theory.

### Scaling Limit of Neural Networks with the Xavier Initialization and Convergence to a Global Minimum

<https://www.arxiv.org/abs/1907.04108>

Analyze the single-layer neural network with tha Xavier initialization in asymptotic regime of large number of hidden units and large numbers of SGD steps. The system can be viewed as stochastic system, analyzed with stochastic analysis, prove that neural network convergences in distribution to a random ODE with a Gaussian distribution, where normalization of Xavier initialization gives completely different result compared to mean-field limit. Due to the limit, optimization problem becomes convex and therefore converges to a global minimum.

### Convergence and Implicit Regularization Properties of Gradient Descent for Deep Residual Networks

<https://www.arxiv.org/abs/2204.07261>

Prove linear convergence of gradient descent to a global minimum, for the training of deep residual network with constant layer width and smooth activation function. Show that trained weight admits a scaling limit which is Hoelder continuous as the depth tends to infinity.

### Provable Convergence of Nesterov's Accelerated Gradient Method for Over-Parameterized Neural Networks

<https://www.arxiv.org/abs/2107.01832>

Analyze NAG in two-layer fully connected network with ReLU activation, show the convergence to global minimum at a non-asymptotic linear rate. Compared to convergence rate of GD, this shows NAG accelerates the training.

### On Feature Learning in Neural Networks with Global Convergence Guarantees

<https://www.arxiv.org/abs/2204.10782>

First show that gradient flow gives linear rate convergence to global minimum when input dimension is no less than the size of training set. Using this fact, show that training second to last layers with GF, prove a linear convergence of network. Also empirically show that unlike in the NTK regime, this model exhibits feature learning.

### Convergence of gradient descent for deep neural networks

<https://www.arxiv.org/abs/2203.16462>

Present a new criterion for convergence of gradient descent to a global minimum, which is provably more powerful than the best available criteria from the literature, the Lojasiewicz inequality.

### Eliminating Sharp Minima from SGD with Truncated Heavy-tailed Noise

<https://www.arxiv.org/abs/2102.04297>

Show that the truncated SGD with heavy-tailed noise eliminate sharp local minima from training trajectory. First, the truncation threshold and the width of the attraction field dictate the order of the first exit time from the associated local minimum. Then, under conditions on loss function, as the learning rate decreases, the dynamics of heavy-tailed truncated SGD resemble the continuous-time Markov chain that never visits any sharp minima.

### Gradient Descent Optimizes Infinite-Depth ReLU Implicit Networks with Linear Widths

<https://www.arxiv.org/abs/2205.07463>

Studies the gradient descent of implicit neural network which has infinitely many layers. Study the convergence of both gradient flow and gradient descnet, and prove a global convergence at a linear rate widht linear widths.

### Topological properties of basins of attraction and expressiveness of width bounded neural networks

<https://www.arxiv.org/abs/2011.04923>

Consider the network with width not exceeding the input dimension, and prove that in this situation the basins of attraction are bounded and their complement cannot have bounded components. Also show that with more conditions, the basins are path-connected.

### Embedding Principle in Depth for the Loss Landscape Analysis of Deep Neural Networks

<https://www.arxiv.org/abs/2205.13283>

Prove an embedding principle in depth, that loss landscape of an NN contains all critical points of the loss landscape of shallower NNs. Propose a critical lifting operator that lift the critical point of a shallow network to critical manifold of the target network, while preserving the outputs which can change the local minimum to strict saddle point. 

### Principal Components Bias in Over-parameterized Linear Models, and its Manifestation in Deep Neural Networks

<https://www.arxiv.org/abs/2105.05553>

Analyze the over-parameterized deep linear neural network, show that in wide enough hidden layers, the convergence rate of parameters is exponentially faster along the directions of the larger principal components of the data, nameing Principal Component bias. 

### Training Two-Layer ReLU Networks with Gradient Descent is Inconsistent

<https://www.arxiv.org/abs/2002.04861>

Show that the training of two-layer ReLU network with gradient descent on a least-squares loss are not consistent, that they only finds a bad local minimum, since it is unable to move the biases far away from their initialization. And in these cases, the network essentially performs linear regression even if the target is nonlinear.

### On Gradient Descent Convergence beyond the Edge of Stability

<https://www.arxiv.org/abs/2206.04172>

Study a local condition for an unstable convergence where the step-size is larger than the admissibility threshold, and establish the global convergence of a two-layer single-neuron ReLU student network aligning with the teacher neuron in a large learning rate.

### The Convergence Rate of Neural Networks for Learned Functions of Different Frequencies

<https://www.arxiv.org/abs/1906.00425>

Study relationship between the frequency of a function and the speed at which a neural network learns it.
Approximate by linear system, and compute eigenfunction which is spherical harmonic functions.
Empirically, theoretically, shallow NN without bias can't learn simple low frequency functions with odd frequencies.

### Understanding How Over-Parameterization Leads to Acceleration: A case of learning a single teacher neuron

<https://www.arxiv.org/abs/2010.01637>

In the setting with single teacher neuron with quadratic activation and over parametrization realized by having multiple student neurons, provably show that over-parameterization helps the gradient descent iteration enter the neighborhood of a global optimal solution.

### Towards Statistical and Computational Complexities of Polyak Step Size Gradient Descent

<https://www.arxiv.org/abs/2110.07810>

Demonstrate that the Polyak step size gradient descent iterates reach a final statistical radius of convergence around the true parameter after logarithmic number of iterations.

### Gradient flow dynamics of shallow ReLU networks for square loss and orthogonal inputs

<https://www.arxiv.org/abs/2206.00939>

Give precise description of the gradient flow dynamics of one-hidden-layer ReLU nets with the mean squared error, and show that it converges to zero loss with the implicit bias towards minimum variation norm.

### Feature Learning in L2-regularized DNNs: Attraction/Repulsion and Sparsity

<https://www.arxiv.org/abs/2205.15809>

Consider the loss surface of DNN with L2-regularization, and show that the loss in terms of the parameters can be reformulated in terms of layerwise activations. So each hidden representations are optimal w.r.t. attraction/repulsion problem and interpolate between the input and output representations, keeping as little information from the input as necessary.

### Support Vectors and Gradient Dynamics of Single-Neuron ReLU Networks

<https://www.arxiv.org/abs/2202.05510>

Examine the gradient flow dynamics in the parameter space when training snigle-neuron ReLU networks, and discover an implicit bias in terms of support vectors. Analyze this gradient flow w.r.t. the magnitude of the norm of initialization, and show that the norm of learned weights strictly increase. Finally prove the global convergence of single ReLU neuron with d=2.

### Convergence of Policy Gradient for Entropy Regularized MDPs with Neural Network Approximation in the Mean-Field Regime

<https://www.arxiv.org/abs/2201.07296>

Show that the softmax policy with shallow NN in a mean-field regime, with infinite-horizon, continuous state and action space, and entropy-regularized MDPs, the objective function increases along the gradient flow. Further, prove that is the regularization is sufficient, the gradient flow converges exponentially fast to the unique stationary solution.

### Neural Network Weights Do Not Converge to Stationary Points: An Invariant Measure Perspective

<https://www.arxiv.org/abs/2110.06256>

Find that the weight of NN do not converge to a stationary points even when the loss stabilizes. Propose a new perspective based on ergodic theory of dynamical system, and study the distribution of weight's dynamics, which converges to an approximate invariant measure.

### The Rate of Convergence of variation-Constrained Deep Neural Networks

<https://www.arxiv.org/abs/2106.12068>

Show that a class of variation-constrained neural network with any width, can achieve near-parameteric rate of convergence n^(-1/2+delta) for an arbitrarily small positive constant, showing that the function space need not to be large as believed.

### Bounding the Width of Neural Networks via Coupled Initialization -- A Worst Case Analysis

<https://www.arxiv.org/abs/2206.12802>

Show that by using same parameter twice for two-layer weight, show that the number of neuron required for convergence can be significantly decreased, for logistic loss and squared loss, implicitly also improving the running time bound also.

### Consistency of Neural Networks with Regularization

<https://www.arxiv.org/abs/2207.01538>

Show that the estimated neural network with regularization converge to true underlying function as the sample size increases, based on method of sieves and the theory on minimal neural networks.

### When does SGD favor flat minima? A quantitative characterization via linear stability

<https://www.arxiv.org/abs/2207.02628>

Prove that if a global minimum is linearly stable for SGD, then the frobenius norm of Hessian should be bounded by batch size and learning rate, otherwise SGD will escape it exponentially fast. 

### Blessing of Nonconvexity in Deep Linear Models: Depth Flattens the Optimization Landscape Around the True Solution

<https://www.arxiv.org/abs/2207.07612>

Study the robust and overparameterized setting of deep linear network, give one negative result and one positive result. On the negative side, show that there is a constant probabiltiy of having a solution corresponding to the ground truth which is neither local or global minimum, however on the positive side, prove that simple sub-gradient method escape from such problematic solution, and converges to a balanced solution that is close to ground truth with flat local landscape.

### Robust Training of Neural Networks Using Scale Invariant Architectures

<https://www.arxiv.org/abs/2202.00980>

Propose an robust training of neural network by modifying it to be scale-invariant, then prove that its convergence only depends on logarithmically on the scale of initialization and loss, where standard SGD may not converge.

### Stochastic Gradient Descent with Exponential Convergence Rates of Expected Classification Errors

<https://www.arxiv.org/abs/1806.05438>

The exponential convergence rate for square loss were shown under a strong low-noise condition, but the expected classification error's exponential convergence is not shown even with low noise condition, only exponential convergence of expected risk. Show an exponential convergence of the classification error in the final phase of ths SGD, for wide classes of differentiable convex loss functions.

### Training Overparametrization Neural Networks in Sublinear Time

<https://www.arxiv.org/abs/2208.04508>

Propose alternative training method of newton types, which gives much faster convergence rate m^(1-a)nd + n^3 for 0.01 < a < 1, m, n, d are each number of parameter, number of data point, and the input dimension. This method relies on the view of neural networks as a set of binary search tree, that each iteration corresponds to modifying a small subset of the nodes.

### The large learning rate phase of deep learning: the catapult mechanism

<https://www.arxiv.org/abs/2003.02218>

Present a class of neural networks with solvable training dynamics, and see two learning rate phase with their phenomena. In small lr regime, the dynamics follow the theory of infinite width, where in large lr regime the convergence to flat minima occurs.

### A Unifying View on Implicit Bias in Training Linear Neural Networks

<https://www.arxiv.org/abs/2010.02501>

Propose a tensor formulation, and characterize the convergence direction as singular vectors, and show that gradient flow finds a stationary point for separable classification, but finds a global minimum for underdetermined regression.

### The staircase property: How hierarchical structure can guide deep learning

<https://www.arxiv.org/abs/2108.10573>

Defines a staircase property for functions over the boolean hypercube, which posits that high-order Fourier coefficients are reachable from low-order Fourier coefficients along increasing chains. Prove that functions with staircase property can be learned in polynomial time using layerwise stochastic coordinate descent on regular neural network. 

### Embedding Principle: a hierarchical structure of loss landscape of deep neural networks

<https://www.arxiv.org/abs/2111.15527>

Prove a general embedding principle of loss landscape of DNNs that unravels a hierarchical structure of the loss landscape of NNs, loss landscape of an NN contains all critical points of all the narrower NNs. Provide a gross estimate of the dimension of critical submanifolds embedded from critical points of narrower NNs. Prove an irreversibility property of any critical embedding.

### Continuous vs. Discrete Optimization of Deep Neural Networks

<https://www.arxiv.org/abs/2107.06608>

Find that the degree of approximation of gradient descent on gradient flow depends on the curvature around the gradient flow trajectory. Show that over DNNs with homogeneous activations, gradient flow trajectories enjoy favorable curvature, that they are well approximated by gradient descent.

### Theoretical insights into the optimization landscape of over-parameterized shallow neural networks

<https://www.arxiv.org/abs/1707.04926>

Show that with quadratic activations, the optimization landscape of training of over-parameterized shallow neural networks has favorable characteristics so that globally optimal models can be found efficiently with various local search heuristics.

### A proof of convergence for stoachastic gradient descent in the training of artificial neural networks with ReLU activation for constant target functions

<https://www.arxiv.org/abs/2104.00277>

Prove that the risk of the SGD process converges to zero if target function is contant.

### Convergence Rates of Training Deep Neural Networks via Alternating Minimization Methods

<https://www.arxiv.org/abs/2208.14318>

Propose a unified framework for analyzing the convergence rate of AM-type training methods, based on j-step sufficient decrease conditions and the Kurdyka-Lojasiewicz property. Show that the detailed local convergence rate if the KL exponent varies in [0, 1).

### Training Scale-Invariant Neural Networks on the Sphere Can Happen in Three Regimes

<https://www.arxiv.org/abs/2209.03695>

Show that there are three regimes of training depending on the effective learning rate value, convergence, chaotic equilibrium, and divergence when training scale-invaritn NNs on the sphere. By studying these regimes in the toy example, show that they have unique features and some specific properties.

### Proxy Convexity: A Unified Framework for the Analysis of Neural Networks Trained by Gradient Descent

<https://www.arxiv.org/abs/2106.013792>

Introduce notion of proxy convexity and proxy Polyak-Lojasiewicz inequalities, which are satisfied if the original objective induces a proxy objective that is implicitly minimized during the gradient. Show that gradient descent on objectives satisfying proxy convexity of proxy PL inequality gives efficient guarantees for proxy objective functions, and many existing guarantees for neural networks can be unified using these two notions.

### Self-Stabilization: The implicit Bias of Gradient Descent at the Edge of Stability

<https://www.arxiv.org/abs/2209.15594>

Demonstrate that the dynamics of gradient descent at the edge of stability can be captured by cubic Taylor expansion. The cubic term causes the curvature to decrease so that stability is restored, named self-stabilization. Self-stabilization makes gradient descent at EoS to follow the projected gradient descent while largest eigenvalue of Hessian is less than 2 / learning rate.

### Plateau in Monotonic Linear Interpolation -- A "Biased" View of Loss Landscape of Deep Networks

<https://www.arxiv.org/abs/2210.01019>

Show that the interpolation over parameter space for both weight and bias gives very different influences on the final output, and there will be a long plateau in the both loss and accuracy interpolation, if different classes have different last-layer biases. This is the phenomena that MLI cannot explain.

### Implicit Bias in Leaky ReLU Networks Trained on High-Dimensional Data

<https://www.arxiv.org/abs/2210.07082>

Consider the two-layer fully-connected neural networks with leaky-ReLU activation. Show that the gradient flow produces a neural network with rank at most two, and is an l2-max-margin solution with linear decision boundary. For the gradient descent, a single step is sufficient to reduce the rank of the network and small rank is preserved throughout training.

### Non-convergence of stochastic gradient descent in the training of deep neural networks

<https://www.arxiv.org/abs/2006.07075>

Show that stochastic gradient descent can fail if depth is much larger than their width, and the number of random initialization does not increase to infinity fast enough.

### The alignment property of SGD noise and how it helps select flat minima: A stability analysis

<https://www.arxiv.org/abs/2207.02628>

Show that if a global minimum is linearly stable for SGD, then it allows to bound the Frobenius norm of Hessian, by learning rate and batch size, otherwise SGD will escape from global minima.

### Learning Ability of Interpolating Convolutional Neural Networks

<https://www.arxiv.org/abs/2210.14184>

Establish the best learning rates of underparamterized DCNNs without parameter restrictions, and show that adding well-defined layer gives interpolating DCNNs with good learning rate of underparamtereized DCNN.

### An initial alignment between neural network and target is needed for gradient descent to learn

<https://www.arxiv.org/abs/2202.12846>

Consider the Boolean target function, and the FCNN with expressive enough activation. Show that without the alignment of initialization and the target function, the learning is impossible with polynomial number of steps and polynomial sized neural network.

### A Convergence Theory for Deep Learning via Over-Parametrization

<https://www.arxiv.org/abs/1811.03962>

When network is overparametrized, SGD can find global minima in polynomial time.

### Gradient Descent Provably Optimizes Over-parametrized Neural Networks

<https://www.arxiv.org/abs/1810.02054>

In two-layer FC ReLU NN, if width is large enough and no two inputs are parallel, gradient descent converges to a globally optimal solution at a linear convergence rate.

### Toward Equation of Motion for Deep Neural Networks: Continuous-time Gradient Descent and Discretization Error Analysis

<https://www.arxiv.org/abs/2210.15898>

Derive and solve an Equation of Motion of DNNs, starting from gradient flow and derive the counter term canceling the discretization error. Then derive continuous differential equation that describes the discrete learning dynamics.

### Improved Overparameterization Bounds for Global Convergence of Stochastic Gradient Descent for Shallow Neural Networks

<https://www.arxiv.org/abs/2201.12052>

Improve the SOTA results in terms of the required hidden layer width. Establish the global convergence of continuous solution of the differential inclusion, and provide relating solutions to the stochastic gradient sequences.

### Convergence to gool non-optimal critical points in the training of neural networks: Gradient descent optimization with one random initialization overcomes all bad non-global local minima with high probability

<https://www.arxiv.org/abs/2212.13111>

Show that in simplified shallow NN with gradient flow overcomes all bad non-global local minima with high probability, and converge to the critical point with risk value close to optimal one. This analysis allow to establish convergence to zero risk value with infinite width.

### A proof of convergence for the gradient descent optimization method with random initialization in the training of neural entworks with ReLU activation for piecewise linear target function

<https://www.arxiv.org/abs/2108.04620>

Prove that the risk of the ANN converges to zero if width, number of initialization, and gradietn step increase to infinity, when target function is piecewise linear and input distribution is uniform distribution on a compact interval. The proof show that the suitable set of global minima form C2 submanifold of parameter space, and the Hessian on this set satisfy maximal rank condition.

### How do noise tails impact on deep ReLU networks?

<https://www.arxiv.org/abs/2203.10418>

Consider the regression of deep ReLU NN under the finite p-th moment noise, and characterize how the optimal rate of convergence depends on p, degree of smoothness, and the intrinsic dimension in a class of nonparametric regression functions with hierarchical composition structure when both the adaptive Huber loss is used. This optimal rate of convergence is not achievable with least square, but can be achieved by the Huber loss with properly chosen parameter adapts to the sample size, smoothness, and moment parameters. Also give concentration inequality for the adaptive estimators.

### On the Convergence of Stochastic Gradient Descent in Low-precision Number Formats

<https://www.arxiv.org/abs/2301.01651>

Study the analysis of SGD for both deterministic and stochastic, obtaining bound that show the effect of number formats.

### Stability Analysis of Sharpness-Aware Minimization

<https://www.arxiv.org/abs/2301.06308>

Demonstrate that SAM dynamics can have convergence instability that occurs near a saddle point. Theoretically prove that saddle point can be an attractor of SAM, and prove that SAM diffusion is worse than vanilla GD in terms of saddle point escape.

### Optimization-Based Separations for Neural Networks

<https://www.arxiv.org/abs/2112.02393>

Prove that under mild assumptions on data distribution, gradient descent can efficiently learn ball indicator functions with depth 2 neural network. Simultaneously show that there are radial distribution on d-dimension data such that ball indicator cannot be learned efficiently by any algorithm better than Omega(d^(-4)), nor by gradient descent to accuracy better than a constant.

### On the Effective Number of Linear Regions in Shallow Univariate ReLU Networks: Convergence Guarantees and Implicit Bias

<https://www.arxiv.org/abs/2205.09072>

Show that when labels are determined by target network with r neurons, GF converges to the case with at most O(r) linear regions implying a generailzation bound.

### On a continuous time model of gradient descent dynamics and instability in deep learning

<https://www.arxiv.org/abs/2302.01952>

Propose the principal flow, which is a continuous time flow that approximates the gradient descent dynamics, admitting the divergence and oscillatory behavior including the local minimia and saddle point escape.

### The Asymmetric Maximum Margin Bias of Quasi-Homogeneous Neural Networks

<https://www.arxiv.org/abs/2210.03820>

Introduce the quasi-homogeneous models that can describe neural networks with homogeneous activations, biases, residual connections, and normalization layers. Genearlize the existing results of maximum-margin bias for those networks, that gradient flow favors a subset of the parameters unlike homogeneous networks, which minimizes asymmetric norm.

### Over-Parametrization Exponentially Slows Down Gradient Descent for Learning a Single Neuron

<https://www.arxiv.org/abs/2302.10034>

For the teacher-student setting, where student has a single ReLU neuron show the global convergence of O(T^{-3}) rate, and also present the lower bound in over-parameterization. This is contrast to the case when teacher also have single neuron, which have exponentially fast convergence rate, which shows that the over-parameterziation can exponentially slow down the convergence.

### Generalization and Stability of Interpolating Neural Networks with Minimal Width

<https://www.arxiv.org/abs/2302.09235>

Show that for logistic-loss minimization, the k-homogeneous shallow neural network has training loss converging to zero at rate O(1/gamma^(2/k) T) where gamma is the margin, with logpolynomial number of neurons. Simultaneously the test loss is bounded by O(1/gamma^(2/k)n), which is contrast the existing stability results whith require polynomial loss and suboptimal generalization rates.

### Phase diagram of training dynamics in deep neural networks: effect of learning rate, depth, and width

<https://www.arxiv.org/abs/2302.12250>

Analzye the maximum eigenvalue of Hessian, show that there are four regimes, eraarly time transitent regime, intermediate saturation regime, progressive sharpening regime, and late time edge of stability regime. 

### On the Training Instability of Shuffling SGD with Batch Normalization

<https://www.arxiv.org/abs/2302.12444>

Show that linear network with batch normalization, trained with single shuffle or random reshuffle SGD, converge to distinct global optimum that are distorted away from gradient descent. SS leads distorted optima and even divergence for classification, but RR avoids both case.

### On the existence of minimizers in shallow residual ReLU neural network optimization landscapes

<https://www.arxiv.org/abs/2302.14690>

Show that for general class of loss function and all continuous target function, the minimum exists for shallow residual ReLU NN. Propose a kind of closure of the search space, and provide criteria for the loss function and underlying probability distribution ensuring that all generalized responses are suboptimal and the minimizer exists.

### Benign Overfitting in Linear Classifiers and Leaky ReLU Networks from KKT Conditions for Margin Maximization

<https://www.arxiv.org/abs/2303.01462>

Show the settings that KKT conditions for margin maximization implies benign overfitting for two-layer Leaky ReLU networks, including noisy class conditional Gaussians. 

### On the existence of optimal shallow feedforward networks with ReLU activation

<https://www.arxiv.org/abs/2303.03950>

Show that there is global minima in the loss landscape of continuous function approximation for two-layer ReLU network. Propose a closure of the search space, so that the minimizers exist, while the functions added by closure perform worse, so the minimizer exist in representable functions.

### A view of mini-batch SGD via generating functions: conditions of convergence, phase transitions, benefit from negative momenta

<https://www.arxiv.org/abs/2206.11124>

Analyze the noise-averaged property of mini-batch SGD for linear model, considering the dynamics of the second moments of model parameters for spectrally expressible approximations. Obtain an explicit expression for the generating function, and find that SGD dynamics convergence regime depend on the spectral distribution, those regimes admit stability condition, and the optimal convergence rate is achieved at negative momenta.

### The Probabilistic Stability of Stochastic Gradient Descent

<https://www.arxiv.org/abs/2303.13093>

Modify the original stability definition to probabilistic stability, and show that only under the lens of probabilistic stability does SGD exhibit rich and pratically relevant phases of learning. These phase diagrams imply that SGD prefers low-rank saddles when the gradient is noisy. Also prove that the probabilistic stability of SGD can be quantified by the Lyapunov exponents of the SGD dynamics.

### Convergence of stochastic gradient descent under a local Lajasiewicz condiiton for deep neural networks

<https://www.arxiv.org/abs/2304.09221>

Show that if initialized in the local region that lajasiewicz condition holds, the SGD converges to a global minimum in the region.

### Pointwise convergence theorem of generalized mini-batch gradient descent in deep neural network

<https://www.arxiv.org/abs/2304.08172>

Restrict the target function to non-smooth indicator functions, and construct a deep neural network inducing pointwise convergence.

### Revisiting Gradient Clipping: Stochastic bias and tight covergence guarantees

<https://www.arxiv.org/abs/2305.01588>

Give the convergence guarantees that is tight for both deterministic and stochastic gradients. Show that the deterministic gradient descent affects higher-order terms of convergence, and the stochastic gradient can not be guaranteed even arbitrary small step sizes.

### Why Learning of Large-Scale Neural Networks Behaves Like Convex Optimization

<https://www.arxiv.org/abs/1903.02140>

Introduce canonical space, and show that the objective of NN is convex in canonical space. Then show that the canonical space and parameter space is related by a pointwise linear transformation, and show that gradient descent methods a.s. convergest to a global minimum of zero loss given that the linear transformation is full rank. 

### Stability of Accuracy of the Training of DNNs Via the Uniform Doubling Condition

<https://www.arxiv.org/abs/2210.08415>

Extend the doubling condition on the training data which ensures the accuracy during training for absolute value activation NN, to uniform doubling condition which do not depends on choices, and extend it to applied to any piecewise lienar functions.

### Stability and Generalization of Stochastic Compositional Gradient Descent Algorithms

<https://www.arxiv.org/abs/230.03357>

Provide the stability and generalization analysis fo stochastic compositional gradient descent algorithms that includes RL, AUC maximization and meta-learning. Define compositional uniform stability and its relation with generalization, and establish uniform stability result for two algorithms SCGD and SCSC.

### Convergence and concentration properties of constant step-size SGD through Markov chains

<https://www.arxiv.org/abs/2306.11497>

Consider the optimization of smooth and strongly convex objective with constant step-size SGD, and study it via Markov chain. Show that with unbiased gradient estiamtes with controlled variance, the iteration converges to an invariatn distribution in TV distance, also the convergence in Wasserstein distance. Then with this invariance property, show that limit distribution holds same concentration properties of gradient, which gives high-confidence bounds.

### Convergence and Stability of the Stochastic Proximal Point Algorithm with Momentum

<https://www.arxiv.org/abs/2111.06171>

Show that SPPAM allows a faster linear convergence to a neighborhood compared to SPPA with better contraction factor, and SPPAM depends on problem constants more favorably, allowing wider range of step size and momentum for convergence.

### Practical Sharpness-Aware Minimization Cannot Converge All the Way to Optima

<https://www.arxiv.org/abs/2306.09850>

Consider SAM with constant perturbation size nd gradietn normalization, find out that SAM has limited capability to global convergence or stationary points. Show that stochastic SAM suffers additive error indicating its convergence towards neighbors, and deterministic SAM also suffer for nonconvex objective.

### Convergence of First-Order Methods for Constrained Nonconvex Optimization with Dependent Data

<https://www.arxiv.org/abs/2203.15797>

Consider stochastic projected gradient under dependent data sampling, for constrained smooth nonconvex optimization. Show that near the stationary point needs t^(-1/4) convergence and complexity eps^(-4) with norm of gradient of Moreau envelope, with only mild mixing condition. This approach is general to derive convergence of adaptive or momentum algorithms.
