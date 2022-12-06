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

Recent works in learned optimizers involve designing algorithms that learn via nested meta-optimization, where the inner loop optimizes the task-level objective and the outer loop learns the optimizer. However there are some issues employing these techniques because of the reliance on unrolled optimization and reinforcement learning component. The authors of this paper refer back to the concept of using checkpoints explored in the community and argue that these checkpoints contain rich information about parameters, metrics, losses and errors. They propose the use of checkpoint datasets instead of large chunk of datasets. The authors have created a dataset consisting of these checkpoints using standard daatsets like CIFAR, MNIST and Cartpole as part of this research. They claim to have included 23 million checkpoints within this dataset. The authors specifically explored generative pre training directly in the parameter space and employed transformer-based diffusion models of neural network parameters. 


#### Dataset for neural network checkpoints

The authors create the checkpoint dataset for the neural network training. They run optimizers like Adam and SGD and generate parameters and record various checkpoints. These checkpoints are augumented to the actual dataset to enable learned optimizer functioning.  Given a checkpoint (θ, l), we
construct augmented tuples (T (θ), l), where T (·) is the parameter-level augmentation. For creating a checkpoint (θ, l), they use a parameter level augumentation T(.) and for them to be valid, they need a fucntion fT (θ)(x) = fθ(x). They make use of permutation augumentation for this pupose. Using thia permutation augumentation, the authors seek to permute the outgoing and incoming weights to preserve the output of the neural network.

#### Generative model for neural network checkpoints

The authors use the dataset generated above to learn parmaters and create learned optimizer over the dataset. For this they propose a generative model that makes use of diffusion that could learn from these given paramters out of checkpoints. They use this diffusion model to learn the given parameters to generate future parameters (noisy). It will be outputting the (θ', l') set where θ' represents the future parameters and l' represents the prompted loss. The diffusion model attempts the signal over loss prediction and the loss would be denoted by:

                                   L(G) = E[||c' − G(θ'j, θ, l', l, j)||]
                                  
The generative diffusion model is a tranformer model and the authors employ the architecture shown in Figure 1 to create a learned optimizer. They use the layered tokenization aproach to pass parameters through the network. This is to accomodate the variable of parameters observed after every layer in a neural network. In each case, the parameters are tokenized into a single token. For layers having really large amounts of parameters (larger networks too), the layers will be flattened first and chuncked into multiple tokens of size M where M is the maximum size of parameters that could be converted to a single token. The scalar values like l and l' are also tokenized and are gifven to the diffusion transformer just as shown in the figure. As shown in figure they employ simple linear per token encoders and per token decoders. The decoders take in the output from the transformer to the future parameter vector and decode it back to its original size. There is no weight sharing between the decoders. There is a global residue connection from the input to the end of the transformer to make sure that the prediction update comes from θ' - θ rather than just the θ' itself. 




## Why G.pt is efficient than other approaches in the community

The paper explores a new direction of utilizing the checkpoint datasets that enable learned optimizer that could effectively learn about the model something everytime it runs through the dataset unlike Adam and SGD optimizers. This paper proposes a simple, scalable and data driven approach which instead of relying on large datasets, makes use of checkpoint datasets unlike other models which tend to rely on unrolled optimizations that could be costly. The following reasons explain why G.pt is most suited for this application. 

1. It is able to rapidly train neural networks from unseen initializations with just one parameter update. 
2. It can generate parameters that achieve a wide range of prompted losses, errors and returns.
3. It is able to generalize to out-of-distribution weight initialization algorithms. 
4. As a generative model, it is able to sample diverse solutions. 
5. It can optimize non-differentiable objectives, such as RL returns or classification errors.

