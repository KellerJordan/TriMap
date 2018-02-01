# TriMap
Experiments and code for TriMap dimensionality reduction

TriMap optimized using sgd with momentum on the full MNIST dataset:

![sgd-momentum](animations/sgd-momentum.gif)

And using vanilla gradient descent:
![sgd](animations/sgd.gif)

Various optimizers on 10k samples:

Initializing by a unit gaussian leads to poorer convergence results
![sgd-N1](animations/sgd_N1.gif)

Better to initialize to 1e-4 * unit gaussian
![sgd-Ne-4](animations/sgd_Ne-4.gif)

Adam reaches a lower loss, but results in several points being "flung" away
![adam](animations/adam.gif)
