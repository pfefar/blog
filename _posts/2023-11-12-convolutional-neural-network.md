---
layout: default
title:  "From Convolution to Neural Network"
subtitle: "Most explanations of CNNs assume the reader understands the convolution operation and how it relates to image processing. I explore convolutions in detail and explain how they are implemented as layers in a neural network."
date:   2023-11-12 13:25:36 -0800
keywords: machine learning, linear algebra, computational mathematics
published: true
---
## Introduction

In learning about convolutional neural networks (CNNs), I read a number of articles, blog posts, and lecture slides, most of which focused on CNNs’ distinct properties such as their architecture, shared weights, or sparse connectivity. My aim is to build up to these properties from ideas that are, at least in my experience, less thoroughly explained: what is a convolution, how is it related to image processing, and how do CNNs implement convolutions? I’ll also mention the benefits of pooling.

### Convolutional neural networks: what and why?

Recall that a neural network (NN) is a hierarchical network of computational units or “neurons” for highly nonlinear function approximation. Each neuron’s input is a set of weighted upstream signals. The neuron treats this set of weighted inputs as a linear combination and optionally applies a nonlinearity.

CNNs are neural networks with a special structure that allows them to efficiently and robustly detect visual patterns in images (LeCun & others, 2015). To see why we might want a specialized neural network, consider a fully connected NN with an input of RGB color images. If each image is 3×256×256 pixels (3 for the number of color channels), then a single neuron in the first fully connected hidden layer would have 196,608 weights. This is a lot. CNNs make a few simplifying assumptions about images and thereby dramatically reduce the number of learnable parameters. In this context, CNNs can be viewed as a kind of regularized NN.


## Convolutions and kernels

As their name suggests, the distinguishing idea behind convolutional neural networks is the convolution, which is related to the idea of a kernel. After understanding convolutions and kernels, the special structure of CNNs will make more sense.

### Discrete convolutions in 1D

A convolution is a mathematical operation on two functions that outputs a function that is a modification of the two inputs. Since it is sufficient for our purposes, I will only discuss the discrete convolution operator, but Goodfellow et al (Goodfellow et al., 2016) has a broader discussion. A discrete convolution between two single-variable functions {% katex %}f{% endkatex %} and {% katex %}g{% endkatex %} , denoted with an asterisk ({% katex %}∗{% endkatex %}), is defined as:

{% katex display %}
(f * g)(x) \triangleq \sum_{i=-\infty}^{\infty} f(i) \cdot g(x-i)
{% endkatex %}

My intuition is that we are “sliding” the function g across the function {% katex %}f{% endkatex %} and outputting a new function in the process. To see this, let’s work through an example. Imagine we have a 1D input signal representing an audio recording with white noise, and we want to eliminate the noise. We can use the convolution operation to do this to do this. Our input signal is a sequence of numbers:

{% katex display %}
f=[10,9,10,9,10,1,10,9,10,8]
{% endkatex %}

To be clear, the function {% katex %}f{% endkatex %} is a mapping from a time point (the index of the list) to a signal value. Using the ordered-pair notation for a function:


{% katex display %}
f={(0,10),(1,9),(2,10),(3,9),(4,10),(5,1),(6,10),(7,9),(8,10),(9,8)}
{% endkatex %}

Here is a plot of the signal:

<div class='figure'>
  <img src="/assets/img_one.png" style="width: 70%; display: block; margin: 0 auto;" alt="descriptive text for image" />
</div>

{% katex display %}
\begin{aligned}
\mathcal{P}\left(\left\{x_i, x_j\right\} \subseteq X\right) & =\left|\begin{array}{ll}
K_{i i} & K_{i j} \\
K_{j i} & K_{j j}
\end{array}\right| \\
& =K_{i i} K_{j j}-K_{i j} K_{j i} \\
& =\mathcal{P}\left(\left\{x_i\right\} \subseteq X\right) \mathcal{P}\left(\left\{x_j\right\} \subseteq X\right)-K_{i j}^2
\tag{1}
\end{aligned}
{% endkatex %}