# Ideas

## Genetic Programming for MAP

There is a SA based MA inference algorithm in Anglican. If so, why not genetic algorithm?

<https://probprog.github.io/anglican/assets/pdf/tolpin-socs-2015.pdf>

The algorithm known as Bayesian Ascent Monte Carlo exactly catches this problem. I think Variational Bayesian Ascent Monte Carlo also works well.

## Towards a liquid type for probabilistic programming

We have various liquid type in traditional programming langauge. Main fallacy in probabilistic programming is that testing is really hard. How can we?

<https://arxiv.org/pdf/2010.07763.pdf>

This paper gives refinement type on the basis of separation logic. Then, can we do similar thing on probabilistic separation logic?

<https://plv.mpi-sws.org/refinedc/paper.pdf>

Maybe following paper contains these things.

<https://arxiv.org/abs/1711.09305>

## Software Testing on probabilistic programming

Software testing works very well for problems as liveness and safety. However, hyperproperties are not that simple, like probability or security.
Can we extend software testing, like importance software testing, so that we can safely test probability-related hyperproperties?

<https://www.cs.cornell.edu/fbs/publications/Hyperproperties.pdf>

### Computability in Probabilistic Programming Language

From the definition of Anglican, it is obvious that ProbProg has either density function or mass function. Can we extend this scope? If not, what random variables can be computed?

<https://danehuang.github.io/papers/compsem.pdf>

<https://arxiv.org/pdf/2101.00956.pdf>
