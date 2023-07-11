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

<https://www.arxiv.org/abs/1802.05296>

Use an explicit and efficient compression, which yields generalization bounds via a simple compression-based framework, and provide some theoretical justification for widespread empirical success in compressing deep nets.

### Generalization Bounds For Meta-Learning: An Information-Theoretical Analysis

<https://www.arxiv.org/abs/2109.14595>

Derive a novel information-theoretic analysis of the generalization property of meta-learning algorithms.

### VC dimension of partially quantized neural networks in the overparameterized regime

<https://www.arxiv.org/abs/2110.02456>

Focus hyperplane arrangement neural networks, and show that HANNs can have VC dimension significantly smaller than the number of weights while being highly expressive.

### Bridging the Gap Between Practice and PAC-Bayes Theory in Few-Shot Meta-Learning

<https://www.arxiv.org/abs/2105.14099>

Relaxing the assumption that distribution of observed task and target task is equal, develop two PAC-Bayes bounds for the few-shot learning setting, thereby bridging the gap between practice and PAC_Bayesian theories.

### A generalization gap estimation for overparameterized models via Langevin functional variance

<https://www.arxiv.org/abs/2112.03660>

Show that a functional variance characterizes the generalization gap even in overparameterized settings. Propose a computationally efficient approximation of the function variance, a Langevin approximation of the functional variance.

### PACMAN: PAC-style bounds accounting for the Mismatch between Accuracy and Negative log-loss

<https://www.arxiv.org/abs/2112.05547>

Introduce an analysis based on point-wise PAC approach over the generalization error accounting mismatch of training loss and test accuracy.

### Differentiable PAC-Bayes Objectives with Partially Aggregated Neural Networks

<https://www.arxiv.org/abs/2006.12228>

Show how averaging over an ensembles of stochastic neural networks enables new partially-aggregated estimators, leading provably lower-varaince to gradietn estimates, and reformulate a PAC-Bayesian bound to derive optimisable differentiable objective. 

### How Much Over-parameterization is Sufficient to Learn Deep ReLU Networks?

<https://www.arxiv.org/abs/1911.12360>

Show the optimization and generalization under polylogarithmic width network w.r.t. n and epsilon^-1.

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

### Dynamics of stochastic gradient descent for two-layer neural networks in the teacher-student setup

<https://www.arxiv.org/abs/1906.08632>

In the teacher-student setup where the student network is trained by data generated from teacher network, show that the dynamics is captured by a set of differential equations, and calculate the generalization error fo student network.

### Improving Generalization of Deep Neural Networks by Leveraging Margin Distribution

<https://www.arxiv.org/abs/1812.10761>

Prove a generalization error bound based on statistics of the entire margin distribution, instead of using minimum margin. 

### Hardness of Noise-Free Learning for Two-Hidden-Layer Neural Networks

<https://www.arxiv.org/abs/2202.05258>

Give superpolynomial statistical query lower bound for learning two-hidden-layer ReLU networks w.r.t. Gaussian input noise free model, where no general SQ lower bound were known for ReLU network of any depth, only having only for adversarial noise or restricted models.

### Learning ReLU networks to high uniform accuracy is intractable

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

<https://www.arxiv.org/abs/2109.14501>

Define and prove the relationship between generalized notions of learnability, and show how this framework is sufficiently general to characterize transfer, multitask, meta, continual, and lifelong learning.

### Ridgeless Interpolation with Shallow ReLU Networks in 1D is Nearest Neighbor Curvature Extrapolation and Provably Generalizes on Lipscitz Functions

<https://www.arxiv.org/abs/2109.12960>

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

### Robust Fine-Tuning of Deep Neural Networks with Hessian-based Generalization Guarantees

<https://www.arxiv.org/abs/2206.02659>

Develop three generalization bound for deep neural networks, depending on the Hessian of parameter for the loss. Each of bound correspond to Vinalla fine-tnuning with or without early stopping, distance-based regularization, and consistent loss with regularization.

### On the Study of Sample Complexity for Polynomial Neural Networks

<https://www.arxiv.org/abs/2207.08896>

Obtain novel results on sample complexity of PNNs, providing some insights in explaining the generalization ability of PNNs.

### Weight Expansion: A New Perspective on Dropout and Generalization

<https://www.arxiv.org/abs/2201.09209>

Introduce the concept of weight expansion, an increase in the signed volume of a parallelotope spanned by the column or row vectors of weight covariance matrix, and show that this is an effective means of increasing the generalization in a PAC-Bayesian setting. Provide that the dropout leads to the weight expansion, and other methods that achieve weight expansion increase generaliation capacity.

### Generalization Bounds for Deep Transfer Learning Using Majority Predictor Accuracy

<https://www.arxiv.org/abs/2209.05709>

Analyze new generalization bound for tranfer learning, which utilize a quantity called majority predictor accuracy. This bound can be computed efficiently from data, so that it can be used as tranferability measure.

### Deep Linear Networks can Benignly Overfit when Shallow Ones Do

<https://www.arxiv.org/abs/2209.09315>

Bound the excess risk of deep linear networks under gradient flow, and show that the randomly initialized deep linear network achieve closely approximate or known bounds for the minimum risk, and with same conditional variance.

### Stability and Generalization Analysis of Gradient Methods for Shallow Neural Networks

<https://www.arxiv.org/abs/2209.09298>

Study the generalization behavior of shallow neural networks via algorithmic stability. For both GD and SGD, develop consistent excess risk bounds by balancing the optimization and generalization via early stopping.

### Periodic Extrapolative Generalisation in Neural Networks

<https://www.arxiv.org/abs/2209.10280>

Formalize the problem of periodic extrapolation, and investigate the generalisation abilities of various architectures. Find that periodic and snake activation functions fail, while traditional sequential models still outperform, bounded by population based training.

### Neural Networks Efficiently Learn Low-Dimensional Representations with SGD

<https://www.arxiv.org/abs/2209.14863>

Prove that the first-layer weight of the NN converge to the k-dimensional principal subspace spanned by true model's basis. This gives generalization bound independent to the width of NN, and outperforming the kernel regime's generalization bound for the degree p polynomials.

### Beyond Lipschitz: Sharp Generalization and Excess Risk Bounds for Full-Batch GD

<https://www.arxiv.org/abs/2204.12446>

Give sharp path-dependent generalization and excess risk guarantees for the full-batch GD. Prove that for nonconvex smooth loss, the full batch GD efficiently generalize close to any stationary point and recovers the generalization error guarantees of stochastic algorithms. For convex smooth loss, show that the generalization error is tigher than the existing bounds for SGD.

### Uniform convergence may be unable to explain generalization in deep learning

<https://www.arxiv.org/abs/1902.04742>

Present examples of overparameterized linear classifiers and neural networks trained by gradient descent where uniform convergence provably cannot explain generalization, even if we take into account the implicit bias of GD. 

### On the Importance of Gradient Norm in PAC-Bayesian Bounds

<https://www.arxiv.org/abs/2210.06143>

Relax uniform bounds assumption by using on-average bounded loss and on-average bounded gradient norm, propose a new generalization bounds exploiting contractivity of the log-Sobolev inequalities. This adds loss-gradient term to the generalization bound which is a surrogate of the model complexity.

### Instance-Dependent Generalization Bounds via Optimal Transport

<https://www.arxiv.org/abs/2211.01258>

Propose a optimal transport interpretation of the generalization problem, which gives instance-dependent generalization bounds depending on the local Lipschitz regularity of prediction function in the data space.

### Overparameterized random feature regression with nearly orthogonal data

<https://www.arxiv.org/abs/2211.06077>

Consider the random feature ridge regression given by two-layer NN, where number of parameter is much larger than the sample size. Establish the concentration of training error, cross-validation, and generalization errors around kernel ridge regression. Then approximate the performance of KRR by a polynomial kernel matrix, finally giving lower bound of the generalization error of RFRR.

### Understanding the Generalization Benefit of Normalization Layers: Sharpness Reductino

<https://www.arxiv.org/abs/2206.07085>

Mathematically analyze the normalization error, showing that it encourages GD to reduce the sharpness of loss surface. In spcific, for the networks with normalization, explain how GD enters EoS regime and characterize the trajectory of GD in this regime, via continuous sharpness reduction flow.

### Norm-based Generalization Bounds for Compositionally Sparse Neural Networks

<https://www.arxiv.org/abs/2301.12033>

Investigate the Rademacher complexity of sparse neural networks, where each neuron receives small number of inputs. These bounds only consider the norm of convolutional filter, not the Toeplitz matrixces. These bound can be better than standard bound, and almost non-vacuous invarious simple classification problems.

### Optimal Learning of Deep Random Networks of Extensive-width

<https://www.arxiv.org/abs/2302.00375>

Derive the asymptotic limit where number of samples, input dmension, and network width are proportiaonally large, derive the Bayesian posterior of outputs and Bayes-optimal test error for regression and classification error.

### Precise Asymptotic Analysis of Deep Random Feature Models

<https://www.arxiv.org/abs/2302.06210>

Prove a universality result for random feature model and deterministic data, that it is equivalent to the deep linear Gaussian model with first two moment matches at each layer. Then use the convex Gaussian min-max theorem to obtain the exact behavior, and characterize the eigendistribution for each layers with showing that depth has tangible effect.

### Koopman-Based Bound for Generalization: New Aspect of Neural Networks Regarding Nonlinear Noise Filtering

<https://www.arxiv.org/abs/2302.05825>

Propose new generalization bound using Koopman operator, which focus the role of final nonlinear transformation, described by the reciprocal of the determinant of the determinant of the weight matrices, and is tighter than existing norm-based bounds. 

### Do Neuarl Networks Generalize from Self-Averaging Sub-classifiers in the Same Way As Adaptive Boosting?

<https://www.arxiv.org/abs/2302.06923>

Show that deep NNs learn a series of boosted classifiers, whose generalization is popularly attributed to self-averaging over an increasing number of interpolating sub-classifiers.

### Efficiently Learning Neural Networks: What Assumptions May Suffice?

<https://www.arxiv.org/abs/2302.07426>

Show that the previous results, that depth-2 network with nondegenerate weight and Gaussian data can be efficiently learned, do not hold for depth-3 network. 

### SGD learning on neural networks: leap complexity and saddle-to-saddle dynamics

<https://www.arxiv.org/abs/2302.11055>

Show that for Gaussian isotropic data and 2-layer neural network trained with SGD, the time complexity of learning function is characterized by complexity measure named as leap, that measures how hierarchical the function are.

### Benign Overfitting for Two-layer ReLU Networks

<https://www.arxiv.org/abs/23023.04145>

Establish the algorithm-dependent risk bound for two-layer ReLU convolution neural network with label flipping noise, and show that NN trained with GD ca achieve near-zero training loss and Bayes optimal test risk.

### Practicality of generalization guarantees for unsupervised domain adaptation with neural networks

<https://www.arxiv.org/abs/2303.08720>

Find that all bounds for unsupervised domain adaptions are vacuous, and that sample generalization terms account for much of the observed looseness, especially when these terms interact with measure of domain shift. Combine recent data-dependent PAC-Bayes analysis, improves the guarantees.

### A generalization gap estimation for overparameterized models via the Langevin functional variance

<https://www.arxiv.org/abs/2112.03660>

Show that a functional variance characterizes the generalization gap in the overparameterized setting, and provide an efficient approximation of it, Langevin functional variance, which only requires the first order gradient of the squared loss.

### Generalization and Stability of Interpolating Neural Networks with Minimal Width

<https://www.arxiv.org/abs/2302.09235>

Consider the scenario that model weights achieve arbitrarily small training error and distance to initialization is small. Show that when the data are separable by NTK, prove that the training loss decays at 1/T with polylogarithmic number of neurons. 

### Operator learning with PCA-Net: upper and lower complexity bounds

<https://www.arxiv.org/abs/2303.16317>

Show the universal approximation by PCA-Net, and derive lower complexity bound that is decaying rate of PCA eigenvalue related to output distribution's complexity, and the other related to inherent complexity of the space of operators. 

### Lipschitzness Effect of a Loss Function on Generalization Performance of Deep Neural Networks Trained by Adam and AdamW Optimizers

<https://www.arxiv.org/abs/2303.16464>

Theoretically prove that the small Lipschitz constant of a loss function gives more uniform stability, which also decreases the generalization performance.

### Generalisation under gradient descent via deterministic PAC-Bayes

<https://www.arxiv.org/abs/2209.02525>

Establish PAC-Bayesian generalization bounds for gradient descent, which also applies to optimisation algorithms that are deterministic. 

### Learning  One-Hidden-Layer ReLU Networks

<https://www.arxiv.org/abs/2304.10524>

Give the polynomial algorithm that learns combination of k ReLU neurons w.r.t. the Gaussian distribution, without any conditions.

### Fine-tuning Neural-Operator architectures for training and generalization

<https://www.arxiv.org/abs/2301.11509>

Design new neural operator sNO+eps, which includes the kernel integral operator as self-attention, and derive upper bound of the Rademacher complexity that do not depend on the norm control of the parameters.

### Learning Theory of Distribution Regression with Neural Networks

<https://www.arxiv.org/abs/2307.03487>

Defines FCNN architecture for space of probability measures, leveraging the approximation theory of functionals. Show that these hypothesis space results almost optimal learning rates for the proposed distribution regression, up to the logarithmic terms via two-stage error decomposition.

### Generalization Guarantees via Algorithm-dependent Rademacher Complexity

<https://www.arxiv.org/abs/2307.02501>

Consider the algorithm-dependent empirical Rademacher complexity, derive novel bounds on finite fractal dimension that extends continuous ones, simplify dimension independent generalization bound for SGD, and recover some VC classes and compression schemes.

### Sparsity-aware generalization theory for deep neural networks

<https://www.arxiv.org/abs/2307.00426>

Present new approach for generalization of deep ReLU net, by taking the sparsity in the hidden layer activation. Show the trade-offs between sparsity and generalization, without making strong assumptions about degree of sparsity. 

### Fast Convergence in Learning Two-Layer Neural Networks with Separable Data

<https://www.arxiv.org/abs/2305.13471>

Prove that two-layer neural network with normalized GD on separable data leads to linear rate of convergence to the global optimum, by showing gradient self-boundedness conditions and log-lipschitz property. Also study generalization via algorithmic-stability analysis, and show that normalized GD does not overfit via finite-time generalization bounds.

### Effective Minkowski Dimension of Deep Nonparametric Regression: Function Approximation and Statistical Theories

<https://www.arxiv.org/abs/2306.14859>

Consider setting where input data is characterized by effective Minkowski dimension, and prove that the sample complexity of deep nonparametric regression only depends on the effective Minkowski dimension. Appply this setting to multivariate Gaussian distribution, and show that effective Minkowski dimension is small for decay rate of eigenvalues of covariance.

### Nonparametric regression using over-parameterized shallow ReLU neural networks

<https://www.arxiv.org/abs/2306.08321>

For Hoelder space of smoothness alpha < (d+3)/2 of d-variate space, prove that shallow neural network with norm constraint are minimax optimal with sufficiently large width, and derive a new size-independent bound for local Rademacher complexity.

### On Certified Generalization in Structured Prediction

<https://www.arxiv.org/abs/2306.09112>

Present novel PAC-Bayesian risk bound for structured prediction where rate of generalization scales with number of examples and their size, assuming taht the data are generated by the Knothe-Rosenblatt rearrangement of a factorizing reference measure.

### A duality framework for generalization analysis of random feature models and two-layer neural networks

<https://www.arxiv.org/abs/2305.05642>

Via duality analysis, show that approximation and estimation in Barron space is equivalent, so that solving one also resolves other problem. Using this, prove that learning of random feature model do not suffer from curse of dimensionality, and derive lower and upper bounds of the minimax estiamtion error with spectrum of kernel.

### Metric Space Magnitude and Generalisation in Neural Networks

<https://www.arxiv.org/abs/2305.05611>

Consider isometry invariant quantity magnitude, and relate its dimension to the generalisation error of neural network.

### Lower Generalizataion Bounds for GD and SGD in Smooth Stochastic Convex Optimization

<https://www.arxiv.org/abs/2303.10758>

Provide excess risk bounds for GD and SGD under non-realizable stochastic convex optimization, showing that existing stability analyses are tight in step size and iteration. Then study the realizable case that optimal solution minimizes all data point, providing lower bound for two cases.

### Theoretical Analysis of Inductive Biases in Deep Convolutional Networks

<https://www.arxiv.org/abs/2305.08404>

Show that approximation of continuous function for CNN requires log d depth where d is input dimensio. Learning sparse function needs log^2 d samples, which indicate its effectiveness on capturing long-range sparse correlation. Then compare to locally-connected network which is version without weight sharing, the CNN requires log^2 d samples while LCN require d samples, and FCNs need d^2 samples.

### Nearly Optimal VC-Dimension and Pseudo-Dimension Bounds for Deep Neural Network Derivatives

<https://www.arxiv.org/abs/2305.08466>

Prove the nearly optimal VC Dim and Pseudo Dim of derivative functions of DNNs, which gives tight approximation result in Sobolev space and characterize the generalization error with loss functions involving function derivative.

### Mind the spikes: Benign overfitting of kernels and neural networks in fixed dimension

<https://www.arxiv.org/abs/2305.14077>

Show that the large derivative of estimator is necessary for benign overfitting, that moderate derivative gives impossible benign overfitting, while spiky smooth kernels give benign overfitting for kernel regression. Also show that ReLU nets can't overfit benignly, and can be solved by adding small high frequency fluctuations.

### From Tempered to Benign Overfitting in ReLU Neura Networks

<https://www.arxiv.org/abs/2305.15141>

Consider 2-layer ReLU NNs, find the transition of overfitting type from tempered at one dimension to benign n high dimensions, showing that input dimension is crucial for type of overfitting.

### Generalization Guarantees of Gradient Descent for Multi-Layer Neural Networks

<https://www.arxiv.org/abs/2305.16891>

Derive the excess risk rate of 1/sqrt(n) for both 2 and 3 layers NN, and give sufficient or necessary conditions of over / under parameterized models. Also show that as the scaling parameter increases or the network complexity decreases, less overparameterization is needed for desired error rate.

### Initialization-Dependent Sample Complexity of Linear Predictors and Neural Networks

<https://www.arxiv.org/abs/2305.16475>

Focus on size-indepdent bound which only consider the Frobenius norm distance from some fixed reference matrix, show the new sample complexity bounds for feed-forward NNs.