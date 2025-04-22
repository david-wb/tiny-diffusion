# Tiny Diffusion

A minimal implementation of a denoising diffusion model for image generation.

# Theory

The "diffusion" process is defined by a markov chain that adds random noise to the image at each step.

```math
x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1 - \alpha_t}\epsilon_{t-1}
```
where 
```math
    \epsilon_t \sim \mathcal{N}(\cdot; 0,  I)
```
and each $\alpha_t$ is a small constant in the range $(0, 1)$.

Observe that
```math
\begin{align*}
x_t &= \sqrt{\alpha_t}x_{t-1} + \sqrt{1 - \alpha_t}\epsilon_t \\
 &= \sqrt{\alpha_t}\sqrt{\alpha_{t-1}}x_{t-2} + \sqrt{\alpha_t}\sqrt{1 - \alpha_{t-1}}\epsilon_{t-2} + \sqrt{1-\alpha_t}\epsilon_{t-1}.
\end{align*}
```
The last two terms can be combined into a single distribution, because
```math
\begin{align*}
\sqrt{\alpha_t}\sqrt{1 - \alpha_{t-1}}\epsilon_{t-2} + \sqrt{1-\alpha_t}\epsilon_{t-1}
&\sim \mathcal{N}(0, \alpha_t(1-\alpha_{t-1},)) + \mathcal{N}(0, (1-\alpha_t)) \\
&\sim \mathcal{N}(0, \alpha_t(1-\alpha_{t-1}) + (1-\alpha_t))) \\
&\sim \mathcal{N}(0, 1 - \alpha_t\alpha_{t-2}) \\
&\sim \sqrt{1 - \alpha_t\alpha_{t-1}}\epsilon_{t-2} 
\end{align*}
```

Thus, $x_t$ can be expressed in closed form as a function of $x_0$ and $\epsilon_0$.
```math
\begin{align*}
x_t &= \sqrt{\alpha_t}x_{t-1} + \sqrt{1 - \alpha_t}\epsilon_t \\
 &= \sqrt{\alpha_t\alpha_{t-1}}x_{t-2} + \sqrt{1 - \alpha_t\alpha_{t-1}}\epsilon_{t-2} \\
 &= \ldots \\
 &= \sqrt{\bar{\alpha}_t}x_{0} + \sqrt{1 - \bar{\alpha}_t}\epsilon_{0} 
\end{align*}
```
where
```math
    \bar{\alpha}_t = \prod_{i=1}^{t}\alpha_i.
```


Thus, the probability of any given state $x_t$ given a starting state $x_0$ is
```math
    q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_{0}, (1 - \bar{\alpha}_t) I).
```


## Reminder: Variational Inference for MLE

Our overal goal is to learn the denoising process. In other words we want to find the optimal parameters $\theta$ to maximize

```math
p_\theta(x_0 | x_t) 
```
for all $x_0, x_t$. 

We can do this by maximizing the overall likelihood of the data
```math
\theta^{*} = \arg\max_{\theta}\mathbb{E}_{x_0}[p_\theta(x_0)].
```


We know from variational inference that

```math
\begin{align*}
\log p_\theta(x_0) &= \text{ELBO} + \text{KL}(q(x_t) || p_\theta(x_t | x_0)) \\
&\ge \text{ELBO}
\end{align*}
```
where 
```math
\begin{align*}
\text{ELBO} &= \mathbb{E}_{x_t\sim q(x_t|x_0)}[\log p_\theta(x_0, x_t)] - \mathbb{E}_{x_t \sim q(x_t|x_0)}[\log q(x_t)] \\
&= \mathbb{E}_{x_t\sim q(x_t|x_0)}[\log \left(p_\theta(x_0 | x_t)p(x_t)\right)] - \mathbb{E}_{x_t \sim q(x_t|x_0)}[q(x_t)] \\
&= \mathbb{E}_{x_t\sim q(x_t|x_0)}[\log p_\theta(x_0 | x_t)] + \mathbb{E}_{x_t\sim q(x_t|x_0)}[\log p(x_t)] - \mathbb{E}_{x_t \sim q(x_t|x_0)}[\log q(x_t)]
\end{align*}
```
The last two terms are constants with respect to $\theta$, so our final objective becomes

```math
\begin{align*}
J(\theta) &= \mathbb{E}_{x_0}\left[\mathbb{E}_{x_t\sim q(x_t|x_0)}[\log p_\theta(x_0 | x_t)]\right]
\end{align*}
```

This means that all we have to do is train a probabilistic model, parameterized by $\theta$, that takes a noisy sample $x_t$ as input, and produces a distribution (mean and variance) over denoised samples $x_0$ output. The next denoised sample can be drawn at random from this output distribution.
<!-- ```math
\begin{align*}
\log p(x) &=\text{KL}(q(z)∣∣p(z∣x)) − \mathbb{E}_{q(z)}\left[\log q(z)− \log p(x,z)
\right] \\
&\ge \mathbb{E}_{q(z)}\left[\log q(z)− \log p(x,z)
\right]
\end{align*}
 ```

Letting $z = x_t$ and $q(z) = p(x_t | x_0)$ we have
```math
\begin{align*}
\log p(x_0) &\ge \mathbb{E}_{p(x_t | x_0)}\left[\log p(x_t | x_0) − \log p(x_0, x_t)\right] \\
&= \mathbb{E}_{p(x_t | x_0)}\left[\log p(x_t | x_0) − \log(p(x_0 | x_t)p(x_t))\right] \\
&= \mathbb{E}_{p(x_t | x_0)}\left[\log p(x_t | x_0) − \log(p(x_0 | x_t)p(x_t))\right] \\
\end{align*}
 ``` -->
