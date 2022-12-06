---
Title: Learning to learn with Generative models of Neural Network Checkpoints
Author: Navya Battula
Date: 30-11-2022
Category: Optimizing neural nets
# heroFullScreen: true
---


The general optimizers used to train neural networks like Adam and SGD are very efficeint in doing their job perfectly. They drive the entire learning process by making convergence happen and model pick some patterns from the given data. But they do suffer from an important draw back of not being able to improve based on past experience. By past experience here we mean that running the same model with the same optimizer isn't going to essentially change the how fast the model is converging. With this being said, the exploration for learned optimizers has been in place for a while in the community. This paper explores a work in that particular direction.

Paper: [Learning to Learn with Generative Models of Neural Network Checkpoints](https://arxiv.org/pdf/2209.12892.pdf) 
Code: https://github.com/wpeebles/G.pt

## Overview

Recent works in learned optimizers involve designing algorithms that learn via nested meta-optimization, where the inner loop optimizes the task-level objective and the outer loop learns the optimizer. However there are some issues employing these techniques because of the reliance on unrolled optimization and reinforcement learning component. The authors of this paper refer back to the concept of using checkpoints explored in the community and argue that these checkpoints contain rich information about parameters, metrics, losses and errors. They propose the use of checkpoint datasets instead of large chunk of datasets. The authors have created a dataset consisting of these checkpoints using standard daatsets like CIFAR, MNIST and Cartpole as part of this research. They claim to have included 23 million checkpoints within this dataset. The authors specifically explored generative pre training directly in the parameter space and employed transformer-based diffusion models of neural network parameters. They claim that their generative modelling technique is better compared to the unrolled optimization and reinforcement learning explored by previous works. 

They express that their model is better for the following reasons:

1. It is able to rapidly train neural networks from unseen initializations with just one parameter update. 
2. It can generate parameters that achieve a wide range of prompted losses, errors and returns.
3. It is able to generalize to out-of-distribution weight initialization algorithms. 
4. As a generative model, it is able to sample diverse solutions. 
5. It can optimize non-differentiable objectives, such as RL returns or classification errors.


### Dataset for neural network checkpoints

The authors create the checkpoint dataset for the neural network training. They run optimizers like Adam and SGD and generate parameters and record various checkpoints. These checkpoints are augumented to the actual dataset to enable learned optimizer functioning.  Given a checkpoint (θ, l), we
construct augmented tuples (T (θ), l), where T (·) is the parameter-level augmentation. For creating a checkpoint (θ, l), they use a parameter level augumentation T(.) and for them to be valid, they need a fucntion fT (θ)(x) = fθ(x). They make use of permutation augumentation for this pupose. Using thia permutation augumentation, the authors seek to permute the outgoing and incoming weights to preserve the output of the neural network.

#### Generative model for neural network checkpoints

The authors use the dataset generated above to learn parmaters and create learned optimizer over the dataset. For this they propose a generative model that makes use of diffusion that could learn from these given paramters out of checkpoints. They use this diffusion model to learn the given parameters to generate future parameters (noisy). It will be outputting the (θ', l') set where θ' represents the future parameters and l' represents the prompted loss. The diffusion model attempts the signal over loss prediction and the loss would be denoted by:

                                   L(G) = E[||θ' − G(θ'j, θ, l', l, j)||]
                                   



