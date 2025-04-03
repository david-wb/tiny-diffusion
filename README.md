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
    p(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_{0}, (1 - \bar{\alpha}_t) I).
```

By Bayes' rule,
```math
    p(x_0 | x_t) = \frac{p(x_t | x_0)p(x_0)}{p(x_t)}
```
Our goal here is to learn a model with parameters $\theta$ such that the log-likelihood of our images $x_0$ is maximized. We know from variational inference that

```math
\begin{align*}
\log p(x) &=\text{KL}(q(z)∣∣p(z∣x)) − \mathbb{E}_{q(z)}\left[\log q(z)− \log p(x,z)
\right] \\
&\ge \mathbb{E}_{q(z)}\left[\log q(z)− \log p(x,z)
\right]
\end{align*}
 ```

Substituting $x=x_0$ and $z = x_t$, we get
```math
\begin{align*}
\log p(x_0) &\ge \mathbb{E}_{q}\left[\log q(x_t)− \log p(x_0, x_t)\right] \\
&\ge \mathbb{E}_{q}\left[\log q(x_t)− \log p(x_0, x_t)
\end{align*}
 ```