# Tiny Diffusion

A minimal implementation of a denoising diffusion model for image generation.

# Theory

The "diffusion" process is defined by a markov chain that adds random noise to the image at each step.

```math
x_t = \sqrt{1 - \beta_t}x_{t-1} + \epsilon_t
```
where 
```math
    \epsilon_t \sim \mathcal{N}(\cdot; 0, \beta_t I)
```
and each $\beta_t$ is a small constant in the range $(0, 1)$.


Equivalently, we can rewrite this as
```math
x_t = \sqrt{1 - \beta_t}x_{t-1} + \sqrt{\beta_t}\epsilon_t
```
where 
```math
    \epsilon_t \sim \mathcal{N}(\cdot; 0, I)
```

Now observe that
```math
\begin{align*}
x_t &= \sqrt{1 - \beta_t}x_{t-1} + \sqrt{\beta_t}\epsilon_t \\
 &= \sqrt{1 - \beta_t}\sqrt{1 - \beta_{t-1}}x_{t-2} + \sqrt{1 - \beta_t}\sqrt{\beta_{t-1}}\epsilon_{t-1} + \sqrt{\beta_t}\epsilon_{t} \\
&= \sqrt{1 - \beta_t}\sqrt{1 - \beta_{t-1}}x_{t-2} + \sqrt{(1 - \beta_t)\beta_{t-1} + \beta_t}\epsilon_{t} \\
&= \sqrt{1 - \beta_t}\sqrt{1 - \beta_{t-1}}x_{t-2} + \sqrt{\beta_{t-1} - \beta_t\beta_{t-1} + \beta_t}\epsilon_{t} \\
&= \cdots \\
&= x_0\prod_{i=1}^{t} \sqrt{1 - \beta_i} + \sum_{i=1}^{t}\prod_{i=1}^{t}
\end{align*}
```
```math

```

Thus
```math
    p(x_t | x_{t-1}) = p(\epsilon_t) = \mathcal{N}(\epsilon_t; 0, \beta_t I) = 
```


The given some image $x_0$ the full joint distrubtion of the diffusion sequence is expressed as 
```math
    p(x_1, \ldots, x_T | x_0) = \prod_{t=1}^{T}p(x_t | x_{t-1}) = \prod_{t=1}^{T}\mathcal{N}(\epsilon_t; 0, \beta_t I)
```
# Implementation

