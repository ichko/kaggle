# Hyper Neural Cellular Automata (HNCA)

This repository contains experiments and ideas around the [fchollet](https://twitter.com/fchollet)'s [ARC Challenge](https://www.kaggle.com/c/abstraction-and-reasoning-challenge/).

## About the challenge

The dataset of this challenge contains a bunch of `tasks`.
A single task consists of a bunch of demonstration input-output pairs - showing
you what kind of algorithm transforms the input to the output. Then each task
has a test input grid and the challenge is to predict the output grid given
the demonstrations.

TODO: Add example grids
_Example Grids_

## Main idea

The idea of my "end-to-end solution" is as follows:

- Encode the demonstration pairs with Multilayer CNN - each pair is encoded into
  a feature vector of fixed size (say 64 dimensions). Average the features
  of the demonstrations of the task.

- Use this vector to infer the parameters of a convolutional cellular-automata that
  is then trained to transform the input pair into the output one.

- We can of the encoding vector as representing an algorithm used to solve the pair.
  Then we use a hyper-network (network used to infer the parameters of another network) to "compile" this vector into an "executable" cellular automaton.
  This inferred cellular automaton is then used to transform the input into the output.

- We can think of the whole process as a meta-learning procedure. We train a network
  that given examples of the task can infer a network that can solve the task.

- The whole network can be trained end-to-end to solve the tasks using gradient
  based optimization (SGD).

## In detail

"Address parameters" in differentiable way to use them in
some differentiable computation.

The ideas here are based on some ideas I had during the making of my masters
thesis project on learning to simulate games with neural networks.
The main idea is to have a _soft addressable computation_. This means
that some learnable parameters of the network are used to be blended
together (addressed) and the resulting tensor is used as
a parameter for computation of the network - similar to what hyper nets do.

- Pick ConvKernels - softmax the feature vector and blend multiple trainable
  kernel banks to infer a single bank used inside the `nca` to solve the task.

- Soft ConvKernels - init tensor of multiple `input conv banks`\* (utility banks).
  Use feature vector to blend all kernels into a single filter.
  Do this _N_ number of times, where _N = output_channels_.
  In this way a single conv bank is computed. This bank is then used inside
  the `nca` to solve the task.
  This procedure leads to better utilization of conv kernels since
  a single kernel can be used multiple times for different output channels.
  More parameter sharing - leads to learning more general conv kernels.
  (since they have to be reused)

_\*input-conv-bank - tensor with dims [input_channels, kernel_size, kernel_size]_

- Attention addressing SoftConvKernel - The addressing in the previous
  bullet used softmax over the feature vector to compute the blending parameters
  for the kernels. This leads to the feature vector being too big (if we want to
  have lots of utility input conv banks).
  Use small vectors to "address" the different conv input banks and
  use attention to blend them. The feature vector has to be used to compute
  multiple keys for the different output channels. (multi-head addressing).

TODO: Add diagrams

## Canonize grids

- Since each grid (regardless if it is input or output) can have a different size
  we canonize the grids by placing each grid in `32x32` grid. Since the max grid size
  is 30 every grid can be placed inside a `32x32` grid.
- The original grid is placed in the center of the canonical grid and
  a white border is placed around the original grid.
- Color unused in the original grids is used to capture the border of the
  original grid. This makes the border of the grid part of the transformation
  that the cellular automaton has to encode - the input border has to be
  transformed into the output one.

TODO: Add examples of canonical grids.
_Example of canonical grid_

## Notes

- NCA infer in latent space - explain in details
