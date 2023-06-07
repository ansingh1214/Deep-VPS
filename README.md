# Deep-VPS
Repository for the work [Variational Deep Learning of Equilibrium Transition Path Ensembles](https://arxiv.org/abs/2302.14857), in which we develop a novel method to infer rates, time-dependent committment probabilities and the importance of reaction descriptors from a reactive trajectory ensemble. 
The repository contains the scripts and the data used for constructing the plots, as well as a Jupyter notebook that illustrates how this method can be applied to a 2D system that was discussed in the paper.
For the illustration, we show how this method can be used to get the time-dependent committment probabilities and the corresponding Doob potential and forces for reactive events:
![viz1](https://github.com/ansingh1214/Deep-VPS/blob/1/anim/viz1.gif)
The form of the loss is given by the difference in Onsager-Machlup action, that provides a natural way to decompose the reactive rate into effective contributions from different degress of freedom. This allows us to infer the importance of different descriptors to the reactive event without making a-priori assumptions
![]()
Moreover, the transient growth of the contributions can be used to infer the the sequence of activation of different modes of the system during a reactive event.
![viz2](https://github.com/ansingh1214/Deep-VPS/blob/1/anim/viz2.gif)
Finally, the learned forces can be used as an optimal control protocol to generate new reactive trajectories. These trajectories are going to be virtually indistinguishable from reference reactive trajectories (see [1](https://doi.org/10.1063/1.5128956) & [2](https://doi.org/10.1103/PhysRevLett.128.028005) for more info).
