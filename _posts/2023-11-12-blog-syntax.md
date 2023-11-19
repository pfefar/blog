---
layout: default
title:  "template syntax"
subtitle: "blog template blocks for math and code"
date:   2023-11-12 13:25:36 -0800
keywords: machine learning, linear algebra, computational mathematics
published: true
---
## Normal Heading

This is a paragrah.This is a paragrah.This is a paragrah.This is a paragrah.

### Discrete convolutions in 1D

{% katex display %}
(f * g)(x) \triangleq \sum_{i=-\infty}^{\infty} f(i) \cdot g(x-i) \tag{1}
{% endkatex %}


### Fundamental Theorem of Calculus
{% katex display %}
\begin{gathered}
\int_a^b f^{\prime}(x) d x=f(b)-f(a) \\
\frac{d}{d x} \int_a^x f(t) d t=f(x)
\end{gathered}
\tag{2}
{% endkatex %}


### a array
{% katex display %}
f=[10,9,10,9,10,1,10,9,10,8] \tag{2}
{% endkatex %}


### a function
{% katex display %}
f={(0,10),(1,9),(2,10),(3,9),(4,10),(5,1),(6,10),(7,9),(8,10),(9,8)} \tag{3}
{% endkatex %}


### Below is a graph block:


Here is a plot of the signal:

<div class='figure'>
  <img src="/assets/images/img_one.png" style="width: 70%; display: block; margin: 0 auto;" alt="descriptive text for image" />
</div>


Below is how to do multiple equstion numbering:
{% katex display %}
\begin{aligned}
\mathcal{P}\left(\left\{x_i, x_j\right\} \subseteq X\right) & =\left|\begin{array}{ll}
K_{i i} & K_{i j} \\
K_{j i} & K_{j j}
\end{array}\right| \\
& =K_{i i} K_{j j}-K_{i j} K_{j i} \\
& =\mathcal{P}\left(\left\{x_i\right\} \subseteq X\right) \mathcal{P}\left(\left\{x_j\right\} \subseteq X\right)-K_{i j}^2
\tag{4}
\end{aligned}
{% endkatex %}

```python
from tinygrad.tensor import Tensor
import tinygrad.nn.optim as optim

class TinyBobNet:
  def __init__(self):
    self.l1 = Tensor.uniform(784, 128)
    self.l2 = Tensor.uniform(128, 10)

  def forward(self, x):
    return x.dot(self.l1).relu().dot(self.l2).log_softmax()

model = TinyBobNet()
optim = optim.SGD([model.l1, model.l2], lr=0.001)

# ... complete data loader here

out = model.forward(x)
loss = out.mul(y).mean()
optim.zero_grad()
loss.backward()
optim.step()

```

{% katex display %}
f=\left[\begin{array}{ccccc}
1 & 2 & 3 & 4 & 5 \\
6 & 7 & 8 & 9 & 10 \\
11 & 12 & 13 & 14 & 15 \\
16 & 17 & 18 & 19 & 20
\end{array}\right] \quad g=\left[\begin{array}{ccc}
a & b & c \\
d & e & f \\
g & h & i
\end{array}\right]
{% endkatex %}


