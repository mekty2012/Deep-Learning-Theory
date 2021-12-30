This repository contains papers and their summary of abstract. I tried to write simple and understandable, but the abstract is sometimes too hard, or not well written. 
Not all papers here are peer-reviewed, so they may contains errors. However if there is comment about conferences the papers are accepted, it will be more credible.

# Conferences

ICLR(International Conference on Learning Represenatation)

ICML(Internal Conference on Machine Learning)

ACML(Asian Conference on Machine Learning

NeurIPS(Neural Information Processing Systems)

AAAI(Association for the Advancement of Artificial Intelligence)

AISTATS(Artificial Intelligence and Statisics Conference)

COLT(Conference on Learning Theory)

EMNLP(Empirical Methods on Natural Language Processing)

CVPR(Computer Vision and Pattern Recognition)

# arxiv

Physis - Disordered Systems and Neural Networks

Mathematics - Probability, Statistics Theory, Numerical Analysis, Information Theory

Compute Science - Artificial Intelligence, Computer Vision and Pattern Recognition, Machine Learning, Neural and Evolutionary Computing

Statistics - Machine Learning

# Terms

## Inductive Bias

In design of neural network, we adds our bias on data to the architectures, like CNN having locally connected structure based on our bias that image is locally correlated. We call such bias as inductive bias, and it can be architectural, loss related, preprocessing, or training algorithm.

## Implicit Bias

If inductive bias is the bias that we intentionally add from our knowledge, implicit bias is bias in coincidence. For example, there are some researches that Neural Networks are more close to simpler functions then complex functions, like the Occum's razor. Such a behavior is not one we intended, we call them implicit bias.

## Generalization

Even if the neural network gives zero error on training dataset, it does not imply neural network will give 100% correct on any data. Such property is called generalization, and people try to explain why neural networks generalize so well, and derive generalization bound, which is a bound of test loss from train loss.

## Memorization

The network, instead of learning the data, it can memorize the data if neural network has too many parameters. It is in general undesirable, because it can leads to low generalization property and overfitting.

## Double Descent (or triple, multiple)

One of foundational researches empirically found that unlike other machine learning models having bias-variance tradeoff, neural network actually have double descent, that increasing number of parameters later decreases the test loss again. It is still not clearly known why such behavior exists, so it is still main open problem.

## Neural Tangent Kernel

One of foundational research found that there exists a neural tangent kernel, that explains the evolution of neural network by kernel gradient descent. This is proven universally for other architectures, and used to some applications and prove properties for neural network. 

## Robustness

Given some test point, if its epsilon neighborhod also have similar output, we say such points are robust. This is one of main task for neural network verification, which assures the network to be safe from the adversarial attack.

## Overparameterization

Many machine learning models like SVM have smaller numbers of parameters, compared to number of data, and such properties are used to derive tight generalization bound for such models. However deep learning models have more number of parameters compared to number of data, so we can't use classical learning theory results to prove generalization bound for neural network. But we have good properties in such overparameterized regime, so overparameterization is one of well used assumptions in deep learning theory.

# Topics

## Infinite Neural Network

There are phenomenons that emerges in the infinite width (or something similar to width) limit, like NNGP correspondence, NTK, etc. People use these properties to understand training dynamics, generalization property, etc.

## Verification

Sometimes we need a checker that assures correctness of neural network. Based on static analysis tools, people create automatic verifier for neural network.

## Physics

People in physics, especially statistical mechanics are experts on dealing with a number of random variables, and neural network is one of them. On the other side, some features of quantum mechanics, like Feynmann diagram and renormalization groups are also helpful on such computation.

## Algebraic Topology

There are understanding of neural network as learning of data manifold, and some literatures argues that algebraic topological properties of such manifold determines the complexity of problem and learnability of network.

## Mean Field Theory

Further on Physics, following mean field assumption, the computation is also becomes simpler, where we can ignore the higher order interactions. 

## Random Matrix Theory

Further on infinite width neural network, people tries to use knowledges and tools from random matrix theory. They mostly find relation between spectrum and generalization property and learnability.

## Learning Theory

With tools from high dimensional probability, people derive generalization bounds for abstract dataset and architectures. This includes computation of complexity measures including VC dimension.

## Complexity Theory

Some of the researches treats the complexity problems related to neural network, or the complexity class that neural network can solve. 

## Loss Landscape

Considers the loss landscape of neural network training, and derive some interesting properties like loss basin, giving intuitions why neural network's training can be simply done with gradient descent (or its variant).

## Invariant Neural Network

Assuming that data have some kind of invariance, formalize such invariance through Lie-group theory, and design neural network that preserves such invariance.
