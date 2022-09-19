### A Convergence Theory for Deep Learning via Over-Parametrization

<https://www.arxiv.org/abs/1811.03962>

When network is overparametrized, SGD can find global minima in polynomial time.

### Gradient Descent Provably Optimizes Over-parametrized Neural Networks

<https://www.arxiv.org/abs/1810.02054>

In two-layer FC ReLU NN, if width is large enough and no two inputs are parallel, gradient descent converges to a globally optimal solution at a linear convergence rate.

## Deep learning generalizes because the parameter-function map is biased towards simple functions

<https://www.arxiv.org/abs/1805.08522>

Using probability-complexity bound from algorithmic information theory, parameter-function map of many DNNs should be exponentially biased towards simple functions.

### Learning Overparametrized Neural Networks via Stochastic Gradient Descent on Structured Data

<https://www.arxiv.org/abs/1808.01204>

Prove that when the data comes from mixtures of well-separated distributions SGD learns a two-layer overparameterized ReLU-network with a small generalization error, even though the network can fit arbitrary labels.

### Fine-Grained Analysis of Optimization and Generalization for Overparameterized Two-Layer Neural Networks

<https://www.arxiv.org/abs/1901.08584>

Analyze training and generalization, give (1) why random label gives slower training (2) generalization bound independent of network size (3) learnability of a broad class of smooth functions.

### Generalization bounds for deep learning

<https://www.arxiv.org/abs/2012.04115>

Introduce desiderata for techniques that predict generalization errors for deep learning models in supervised learning. Focuse on generalization error upper bound, and inttroduce a categorisation of bounds depending on assumptions on the algorithm and data. 

### Stronger generalization bounds for deep nets via a compression approach

<https://arxiv.org/abs/1802.05296>

Use an explicit and efficient compression, which yields generalization bounds via a simple compression-based framework, and provide some theoretical justification for widespread empirical success in compressing deep nets.

### Generalization Bounds For Meta-Learning: An Information-Theoretical Analysis

<https://arxiv.org/abs/2109.14595>

Derive a novel information-theoretic analysis of the generalization property of meta-learning algorithms.

### VC dimension of partially quantized neural networks in the overparameterized regime

<https://arxiv.org/abs/2110.02456>

Focus hyperplane arrangement neural networks, and show that HANNs can have VC dimension significantly smaller than the number of weights while being highly expressive.

### Bridging the Gap Between Practice and PAC-Bayes Theory in Few-Shot Meta-Learning

<https://arxiv.org/abs/2105.14099>

Relaxing the assumption that distribution of observed task and target task is equal, develop two PAC-Bayes bounds for the few-shot learning setting, thereby bridging the gap between practice and PAC_Bayesian theories.

### A generalization gap estimation for overparameterized models via Langevin functional variance

<https://arxiv.org/abs/2112.03660>

Show that a functional variance characterizes the generalization gap even in overparameterized settings. Propose a computationally efficient approximation of the function variance, a Langevin approximation of the functional variance.

### Estimates on the generalization error of Physics Informed Neural Networks (PINNs) for approximating PDEs

<https://www.arxiv.org/abs/2006.16144>

Provide upper bound on the generalization error of PINNs approximating solutions of the forward problem for PDEs.

### PACMAN: PAC-style bounds accounting for the Mismatch between Accuracy and Negative log-loss

<https://arxiv.org/abs/2112.05547>

Introduce an analysis based on point-wise PAC approach over the generalization error accounting mismatch of training loss and test accuracy.

### Differentiable PAC-Bayes Objectives with Partially Aggregated Neural Networks

<https://arxiv.org/abs/2006.12228>

Show how averaging over an ensembles of stochastic neural networks enables new partially-aggregated estimators, leading provably lower-varaince to gradietn estimates, and reformulate a PAC-Bayesian bound to derive optimisable differentiable objective. 

### How Much Over-parameterization is Sufficient to Learn Deep ReLU Networks?

<https://www.arxiv.org/abs/1911.12360>

Show the optimization and generalization under polylogarithmic width network w.r.t. n and epsilon^-1.

### Approximation bounds for norm constrained neural networks with applications to regression and GANs

<https://www.arxiv.org/abs/2201.09418>

Prove approximation capacity of ReLU NN with norm constraint on the weights, especially upper and lower bound of approximation error of smooth function class, where lower bound comes from Rademacher complexity. Using this bounds, analyze convergence of regression and distribution estimation by GANs.

### Weight Expansion: A New Perspective on Dropout and Generalization

<https://www.arxiv.org/abs/2201.09209>

Define weight expansion which is the signed volume of a parallelotope spanned by column or row vectors of the weight covariance matrix, show that weight expansion is an effective means of increasing the generalization in a PAC Bayesian setting, and prove that dropout leads to weight expansion.

### Generalization Error Bounds on Deep Learning with Markov Datasets

<https://www.arxiv.org/abs/2201.11059>

Derive upper bounds on generalization errors for deep NNs with Markov datasets, based on Koltchinskii and Panchenko's approach for bounding the generalization error of combined classifiers. 


### Algorithm-Dependent Generalization Bounds for Overparameterized Deep Residual Networks

<https://www.arxiv.org/abs/1910.02934>

Show that gradient descent's solution consistutes small subset of entire function class, however is sufficiently large to guarantee small training error. Also gives generalization gap that is logarithmic to depth.

### Non-Vacuous Generalisation Bounds for Shallow Neural Networks

<https://www.arxiv.org/abs/2202.01627>

Derive new generalization bound for shallow neural network with L2 normalization of data, and erf or GELU activation, based on PAC-Bayesian theory.

### Support Vectors and Gradient Dynamics for Implicit Bias in ReLU Networks

<https://www.arxiv.org/abs/2202.05510>

Examine gradient flow dynamics in the parameter space when training single-neuron ReLU networks, with implicit bias that support vectors plays a key role in why and how ReLU networks generalize well.

### The Sample Complexity of One-Hidden-Layer Neural Networks

<https://www.arxiv.org/abs/2202.06233>

Study norm-based uniform convergence bound for neural networks, proving that controlling spectral norm of weight matrix is insufficient for uniform cnovergence guarantee, where stronger Frobenius norm control is sufficient. Also show that when activation is sufficiently smooth and some convolutional networks, spectral norm is sufficient.

### Benign Overfitting without Linearity: Neural Network Classifiers Trained by Gradient Descent for Noisy Linear Data

<https://www.arxiv.org/abs/2202.05928>

Consider the generalization error of two-layer neurla networks trained to interpolation by gradient descent on the logistic loss. Assuming that data has well-separated class-conditional log-concave distribution with adversary corruption, show that neural network can perfectly fit noisy training label with test error close to Bayes-optimal error. 

### Random Feature Amplification: Feature Learning and Generalization in Neural Networks

<https://www.arxiv.org/abs/2202.07626>

For the XOR-like data distribution, show that two-layer ReLU network trained by gradient descent achieve generalization error close to the label noise late. Use a proof technique that at initialization, majority of neurons are weakly correlated with useful features, where gradient descent amplify these weak features to strong, useful features.

### On Measuring Excess Capacity in Neural Networks

<https://www.arxiv.org/abs/2202.08070>

Studies excess capacity, that is given a capacity measure like Rademacher complexity, how much can we constrain hypothesis class while maintaining comparable empirical error. Extend existing generalization bound to accomodate composition, addition, and convolution, to give measure related to Lipschitz constant of layers and (2, 1) group norm distance to initialization. Show that this measure can be kept small, and since excess capacity increases with task difficulty, this points towards an unnecessarily large capacity of unconstrained models.

### The merged-staircase property: a necessary and nearly sufficient condition for SGD learning of sparse functions on two-layer neural networks

<https://www.arxiv.org/abs/2202.08658>

Study O(d) sample complexitiy in large ambient dimension d, of data with binary input which depend on latent low dimensional subspace, and two-layer network in mean-field regime, characterizes a hierarchical property "merged-staircase property" which is necessary and nearly sufficient. Show that non-linear training is necessary, that NTK-like learning is inefficient.

### On PAC-Bayesian reconstruction guarantees for VAEs

<https://www.arxiv.org/abs/2202.11455>

Analyze the VAE's reconstruction ability for unseen test data with PAC-Bayes theory. Provide generalisation bounds on the theoretical reconstruction error and provide insights on the regularisation effect of VAE objective.

### KAM Theory Meets Statistical Learning Theory: Hamiltonian Neural Networks with Non-Zero Training Loss

<https://www.arxiv.org/abs/2102.11923>

Hamiltonian neural network, which approximates the Hamiltonian with neural network, is perturbation from the true dynamics under non-zero loss. To apply perturbation theory for this, called KAM theory, provide a generalization error bound for Hamiltonian neural networks by deriving an estimate of the covering number of the gradient of the MLP, then giving L infinity bound on the Hamiltonian.

### Dynamics of stochastic gradient descent for two-layer neural networks in the teacher-student setup

<https://www.arxiv.org/abs/1906.08632>

In the teacher-student setup where the student network is trained by data generated from teacher network, show that the dynamics is captured by a set of differential equations, and calculate the generalization error fo student network.


### Improving Generalization of Deep Neural Networks by Leveraging Margin Distribution

<https://www.arxiv.org/abs/1812.10761>

Prove a generalization error bound based on statistics of the entire margin distribution, instead of using minimum margin. 


### Haredness of Noise-Free Learning for Two-Hidden-Layer Neural Networks

<https://www.arxiv.org/abs/2202.05258>

Give superpolynomial statistical query lower bound for learning two-hidden-layer ReLU networks w.r.t. Gaussian input noise free model, where no general SQ lower bound were known for ReLU network of any depth, only having only for adversarial noise or restricted models.

### Analyzing Lottery ticket Hypothesis from PAC-Bayesian Theory Perspective

<https://www.arxiv.org/abs/2205.07320>

Hypothesize that the 'winning tickets' have relatively sharp minima, which is a disadvantage in terms of generalization ability, and confirm this hypothesis with PAC-Bayesian theory. Find that the flatness is useful for improving the accuracy and robustness to label noise, and the distance from the initial weights is deeply involved in winning tickets.

### Training ReLU networks to high uniform accuracy is intractable

<https://www.arxiv.org/abs/2205.13531>

Quantify the number of training samples needed for any training algorithm to guarantee a given uniform accuracy on any learning problem of ReLU neural networks. Prove that the minimal number of training samples scales exponentially both in the depth and the input dimension, showing that uniform accuracy is intractable.

### The Interplay Between Implicit Bias of Benign Overfitting in Two-Layer Linear Networks

<https://www.arxiv.org/abs/2108.11489>

Derive bounds on the excess risk when the covariates satisfy sub-Gaussianity and anti-concentration properties with sub-Gaussian, independent noise, with the two layer linear neural network trained with L2 loss, gradient fow. The bound emphasize the role of both the quality of the initialization and the properties of the data covariance matrix.

### Learning to Reason with Neural Networks: Genearlization, Unseen Data and Boolean Measures

<https://www.arxiv.org/abs/2205.13647>

Show that for learning logical functions with GD on symmetric NNs, the generalization error can be lower bounded in terms of the noise-stability of target function. And show for the distribution shift setting, the generalization error of GD admits tight characterization in terms of the Boolean influence.

### Excess Risk of Two-Layer ReLU Neural Networks in Teacher-Student Settings and its Superiority to Kernel Methods

<https://www.arxiv.org/abs/2205.14818>

Consider the excess risk of two-layer ReLU neural networks in a teacher-student regression models, in which a student network learns a unknown teacher network through its outputs. Show that the student network provably reaches a near-global optimal solution and outperforms any kernel methods estimator.

### A general approximation lower bound in Lp norm, with applications to feed-forward neural networks

<https://www.arxiv.org/abs/2206.04360>

Prove a general lower bound on how well function set F can be approximated in Lp norm by another function set G, in terms of packing number, range of F and the fat-shattering dimnesion of G. Instantiate this bound to the case when G is piecewise-polynomial feed-forward neural network.

### Trajectory-dependent Generalization Bounds for Deep Neural Networks via Fractional Brownian Motion

<https://www.arxiv.org/abs/2206.04359>

Characterize the SGD recursion via a SDE by assuming the incurred stochastic gradient noise follows the fractional Brownian motion. Then identify the Rademacher complexity in terms of the covering numbers and relate it to Hausdorff dimension of the optimization trajectory, and derive a novel generalization bound for deep neural network.

### On the Generalization Power of the Overfitted Three-Layer Neural Tangent Kernel Model

<https://www.arxiv.org/abs/2206.02047>

Show that for some function set, the test error of the overfitted three-layer NTK is upper bounded by an expression that decreases with the number of neurons of the two hidden layers. 

### Robust Fine-Tuning of Deep Neural Networks with Hessian-based Generalization Guarantees

<https://www.arxiv.org/abs/2206.02659>

Develop the Hessian distance-based generalization bounds for a wide range of fine-tuning methods. Design an algorithm that incorporates consistenct losses and distance-based regularization for fine-tuning. Also prove a generalization error bound of algorithm under class conditional independent noise in the training dataset labels.

### Regularization Matters: Generalization and Optimization of Neural Nets v.s. their Induced Kernel

<https://www.arxiv.org/abs/1810.05369>

Sample efficiency depend on the presence of the regularizer, that regularized NN requires O(d) samples but the NTK requires O(d^2) samples. 

### Limitations of Lazy Training of Two-layers Neural Networks

<https://www.arxiv.org/abs/1906.08899>

In two-layers neural networks with quadratic activations, (RF) train only last layer (NT) linearized dynamics (NN) full training has unbounded gap of prediction risk.

### Neural Networks can Learn Representations with Gradient Descent

<https://www.arxiv.org/abs/2206.15144>

Consider the problem of learning the solution g(Ux) with U projects to lower dimension, show that the gradient descent learns a representation depending only on the directions relevant to solution, with improved sample complexity compared to kernel regime. Also show that transfer learning setup with same U, show that popular heuristic has target complexity independent to input dimension.

### Learning and generalization of one-hidden-layer neural networks, goind beyong standard Gaussian data

<https://www.arxiv.org/abs/2207.03615>

Analyzes the convergence and generalization of shallow NN, when the input features follow finite mixture of Gaussians, with the labels are generated from teacher model.

### Towards understanding how momentum improves generalization in deep learning

<https://www.arxiv.org/abs/2207.05931>

Formally study how momentum improves generalization, on a binary classification setting where a one-hidden layer CNN trained with GD+M provably generalizes better than the same network trained with GD. Key insight is that, momentum is beneficial in datasets where the examples share some feature but differ in their margin.

### On the Study of Sample Complexity for Polynomial Neural Networks

<https://www.arxiv.org/abs/2207.08896>

Focus on depth-independent bound on sample complexity of general classes of polynomial neural networks, especially using the square function as activation. This is much easier to analyze due to its structure, while the generalization ability is not still well known.

### On generalization bonds for deep networks based on loss surface implicit regularization

<https://www.arxiv.org/abs/2201.04545>

Study how local geometry of loss landscape around local minima affects the statistical properties of SGD with noise. Under reasonable assumptions, this geometry forces SGD to stay close to a low dimensional subspace, which gives implicit regularization and tighter bound on the generalization error for DNNs. Then derive the bound on the spectral norm of weights instead of number of network parameter, when stagnation occur.

## Deep Learning and the Information Bottleneck Principle

<https://www.arxiv.org/abs/1503.02406>

Show that any DNN can be quantified by the mutual information between the layers and the input and output variables, and calculate the optimal information theoretical limits of the DNN and obtain finite sample generalization bounds.

### Towards a theory of out-of-distribution learning

<https://arxiv.org/abs/2109.14501>

Define and prove the relationship between generalized notions of learnability, and show how this framework is sufficiently general to characterize transfer, multitask, meta, continual, and lifelong learning.

### Ridgeless Interpolation with Shallow ReLU Networks in 1D is Nearest Neighbor Curvature Extrapolation and Provably Generalizes on Lipscitz Functions

<https://arxiv.org/abs/2109.12960>

Prove a precise geometric description of all one layer ReLU networks with a single linear unit, with single input/output dimension, which interpolates a given dataset. Also show that ridgeless ReLU interpolants achieve the best possible generalization for learning 1d Lipscitz functions, up to universal constants.

### Superior generalization of smaller models in the presence of significant label noise

<https://www.arxiv.org/abs/2208.08003>

Find that under the mislabeled examples, increasing the network size can be harmful, so that th best generalization is achieved by some model with intermediate size.

### On the generalization of learning algorithms that do not converge

<https://www.arxiv.org/abs/2208.07951>

Focus on the generalization of neural networks whose training dynamics do not necessarily converges to fixed points. Propose statistical algorithmic stability, then prove that the stability of time-asymptotic behavior relates to its generalization.

### Sample Complexity of Offline Reinforcement Learning with Deep ReLU Networks

<https://www.arxiv.org/abs/2103.06671>

Study the statistical theory of offline RL with deep ReLU network approximation, which depends on measure of distributional shift, dimension of state-action space, and the smoothness parameter of the MDP. This complexity holds under two consideration, Besov dynamic closure and the correlated structure from value regression.

### An initial alignment between neural network and target is needed for gradient descent to learn

<https://www.arxiv.org/abs/2202.12846>

Consider the Boolean target function, and the FCNN with expressive enough activation. Show that without the alignment of initialization and the target function, the learning is impossible with polynomial number of steps and polynomial sized neural network.

### Robust Fine-Tuning of Deep Neural Networks with Hessian-based Generalization Guarantees

<https://www.arxiv.org/abs/2206.02659>

Develop three generalization bound for deep neural networks, depending on the Hessian of parameter for the loss. Each of bound correspond to Vinalla fine-tnuning with or without early stopping, distance-based regularization, and consistent loss with regularization.

### On the Study of Sample Complexity for Polynomial Neural Networks

<https://www.arxiv.org/abs/2207.08896>

Obtain novel results on sample complexity of PNNs, providing some insights in explaining the generalization ability of PNNs.

### Weight Expansion: A New Perspective on Dropout and Generalization

<https://www.arxiv.org/abs/2201.09209>

Introduce the concept of weight expansion, an increase in the signed volume of a parallelotope spanned by the column or row vectors of weight covariance matrix, and show that this is an effective means of increasing the generalization in a PAC-Bayesian setting. Provide that the dropout leads to the weight expansion, and other methods that achieve weight expansion increase generaliation capacity.

### Bounding The Rademacher Complexity of Fourier Neural Operator

<https://arxiv.org/abs/2209.05150>

Investigate the bounding of Rademacher complexity of FNO based on some group norms, and the generalization error of the FNO models. 

### Generalization Bounds for Deep Transfer Learning Using Majority Predictor Accuracy

<https://arxiv.org/abs/2209.05709>

Analyze new generalization bound for tranfer learning, which utilize a quantity called majority predictor accuracy. This bound can be computed efficiently from data, so that it can be used as tranferability measure.