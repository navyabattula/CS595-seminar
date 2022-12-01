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

## Introduction

Recent works in learned optimizers involve designing algorithms that learn via nested meta-optimization, where the inner loop optimizes the task-level objective and the outer loop learns the optimizer. However there are some issues employing these techniques because of the reliance on unrolled optimization and reinforcement learning component. The authors of this paper refer back to the concept of using checkpoints explored in the community and argue that these checkpoints contain rich information about parameters, metrics, losses and errors. They propose the use of checkpoint datasets instead of large chunk of datasets. The authors have created a dataset consisting of these checkpoints using standard daatsets like CIFAR, MNIST and Cartpole as part of this research. They claim to have included 23 million checkpoints within this dataset. The authors specifically explored generative pre training directly in the parameter space and employed transformer-based diffusion models of neural network parameters. They claim that their generative modelling technique is better compared to the unrolled optimization and reinforcement learning explored by previous works. 

They express that their model is better for the following reasons:

First, it is able to rapidly train neural networks from unseen initializations with just one parameter update (Figure 3). 
Second, it can generate parameters that achieve a wide range of prompted losses, errors and returns (Figure 5).
Third, it is able to generalize to out-of-distribution weight initialization algorithms (Figure 6). 
Fourth, as a generative model, it is able to sample diverse solutions (Figure 8). 
Finally, it can optimize non-differentiable objectives, such as RL returns or classification errors.

## Generative pre training for neural checkpoint

### Dataset for neural network checkpoints






