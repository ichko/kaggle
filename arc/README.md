# Hyper Neural Cellular Automata (HNCA)

This repository contains experiments and ideas around the [fchollet](https://twitter.com/fchollet)'s [ARC Challenge](https://www.kaggle.com/c/abstraction-and-reasoning-challenge/).

Presentation of the model - [Google Slides](https://docs.google.com/presentation/d/1kr_g2416fnWQQePeG7R_mS0YjO5J-_y1KNVEGBNrvbg/edit?usp=sharing)

Run training with

```sh
wandb local # Or log in wandb

python -m src.main --config conv_programmable_nca --from-scratch
```

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

- "Programmer - Solver" meta learning model.

- Encode the demonstration pairs with Multilayer CNN - each pair is encoded into
  a feature vector of fixed size (say 64 dimensions). Average the features
  of the demonstrations of the task.

- Use this vector to infer the parameters of a convolutional cellular-automata that
  is then trained to transform the input pair into the output one.

- We can think of the encoding vector as representing an algorithm used to solve the pair.
  Then we use a hyper-network (network used to infer the parameters of another network) to "compile" this vector into an "executable" cellular automaton.
  This inferred cellular automaton is then used to transform the input into the output.

- We can think of the whole process as a meta-learning procedure. We train a network
  that given a demonstrations of the task can infer a network that can solve the task.

- The whole network can be trained end-to-end to solve the tasks using gradient
  based optimization (SGD).

---

- The feature extracting CNN and the hyper-network form the programmer. A network
  that programs another network.

- The solver is the CA. The parameters of the CA are runtime activations.

## In detail

"How do you infer a vector long enough to contain the parameters of the solver?"

"Learn a banks of parameters and address them in differentiable way" (_soft addressable computation_)

The idea is somewhat inspired by learning assets to simulate games from my masters theses.
In the context of learning parameters of networks we will have the following setup.
We will initialize a bank of tensors, and the use the features of the programmer
network to blend the parameters in the bank and infer the runtime parameters.
This setup utilizes representations to come up with small useful set of
"utility" tensors, that are then combined runtime to produce the final solver.

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

- Attention addressing SoftConvKernel - The addressing in the previous bullet used softmax over the feature vector to compute the blending parameters for the kernels. This leads to the feature vector being too big (if we want to have lots of utility input conv banks). Use small vectors as "addresses" of the parameters in the bank. Use attention to blend them. The feature vector has to be used to compute multiple keys for the different output channels. (multi-head addressing).

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

- TODO: Save demo gifs in separate thread

- NCA infer in latent space - explain in details

**Things to try**

- [ ] Give the network the ability to use global context

  - The network must be able to see important information from the
    input grid at all times (maybe encoded in all the latent channels)
  - [ ] Use the demonstration pairs to infer a network that using the
        test input infers the solver network

- [ ] Iterative building of parameter banks

  - Use a bank of 2D params to build a 3D bank
  - Use the bank of 3D params to build the hyper conv layers

- Data augmentation

  - [ ] Color permutations
  - [ ] Symmetries

- "Programmer" network ideas (the network that infers the parameters of the "solver" network)

  - [x] Linear programmer
  - Embed the task
    - [x] Single vector representation (used to infer addresses)
    - [ ] Embed the whole network (the addresses)
  - [ ] RNN Programmer
  - [ ] Tree recursion programmer
    - Network that takes a list of fixed length 1D tensors and duplicates then
    - Use it to rapidly build many addresses
  - [ ] Transformer
  - [ ]

- "Solved" network ideas
  - [x] Currently experimenting with NCA
  - [x] Some of the channels in the "core" of the CNN-RNN are latent
    - Mind the way the activation at the end is done
  - [x] Embedded NCA - embed the actual grids (transform only the channels in order not to lose spatial information)
    - Did not get good results
    - [ ] Revisit
  - [ ] CNN Directly solving the pattern
  - [ ] Transformer
