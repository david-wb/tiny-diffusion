# Tiny Diffusion

A minimal implementation of a denoising diffusion model for image generation.

# Diffusion Process

The "diffusion" process is a Markov chain that adds Gaussian noise to a given image $x_0$ at each step. The posterior distribution of the
diffused image samples are given by

```math
q(x_1, \ldots, x_t \mid  x_0) = \prod_{i=1}^{t}q(x_i \mid  x_{i-1}).
```

The transition probabilities are Gaussians

```math
q(x_t \mid  x_{t-1}) = \mathcal{N}(x_{t}; \sqrt{\alpha_{t}} x_{t-1}, (1 - \alpha_t) I).
```

We can equivalently write

```math
x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1 - \alpha_t}\epsilon_{t-1}
```

where each $\epsilon_t \sim \mathcal{N}(\cdot; 0,  I)$ is unit noise.

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
    q(x_t \mid  x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_{0}, (1 - \bar{\alpha}_t) I).
```

# Denoising Process

The denoising process, also called the "reverse" process, is modeled by another Markov chain which iteratively removes noise from a sample $x_t$ to recover the original image $x_0$:

```math
p_\theta(x_0, x_1, \ldots, x_t) = p(x_t)\prod_{i=0}^{t}p_\theta(x_{i-1}\mid x_i)
```

The starting point $x_t$ is assumed to be random Gaussian noise:

```math
    p(x_t) = \mathcal{N}(x_t; 0, I).
```

The transitions are learned Gaussians parameterized by $\theta$:

```math
p_\theta(x_{i-1}\mid x_i) = \mathcal{N}(x_{i-1}; \mu_{\theta}(x_i, i), \Sigma_{\theta}(x_i, i)).
```

The functions $\mu_\theta(x_i, i)$ and $\Sigma_{\theta}(x_i, i)$ represent a parametric model, typically a neural network, which takes as inputs an image $x_i$ and the associated timestep $i$, and outputs the mean and variance of a Gaussian distribution for the denoised sample $x_{i-1}$. The objective is to learn the parameters $\theta$ of this model.

## Reminder: Variational Inference for MLE

As we will see, we can learn the parameters $\theta$ by maximising the likelihood of the data

```math
\mathbb{E}_{x_0}[p_\theta(x_0)] = \mathbb{E}_{x_0}\left[\int p_\theta(x_0, x_1, \ldots x_t) d\mathbf{x}_{1:t}\right]
```

Our training objective is thus to find the optional paraemters $\theta^{*}$:

```math
\theta^{*} = \arg\max_{\theta}\mathbb{E}_{x_0}[p_\theta(x_0)].
```

Letting $z = \{x_1, ... ,x_t\}$ for brevity, we know from variational inference that

```math
\begin{align*}
\log p_\theta(x_0) &= \text{ELBO} + \text{KL}(q(z\mid x_0) \mid \mid  p_\theta(z \mid  x_0)) \\
&\ge \text{ELBO}
\end{align*}
```

Maximizing ELBO increases that data likelihood _and_ makes $p_\theta(z \mid  x_0)$ closer to the true posterior $q(z\mid x_0)$.
where

```math
\begin{align*}
\text{ELBO} &= \mathbb{E}_{z \sim q(\cdot\mid x_0)}\left[\log \frac{p_\theta(x_0, z)}{q(z\mid x_0)}\right]
\end{align*}
```

Expanding the term inside the expectation we get

```math
\begin{align*}
\log \frac{p_\theta(x_0, z)}{q(z\mid x_0)}
&= \log \frac{ \left(p(x_t)\prod_{i=1}^{t}p_\theta(x_{i-1}\mid x_i) \right)}{q(z\mid x_0)} & \\
&= \log \frac{ \left(p(x_t)\prod_{i=1}^{t}p_\theta(x_{i-1}\mid x_i) \right)}{\prod_{i=1}^{t}q(x_i \mid  x_{i-1})} & \\
&= \log p(x_t) + \sum_{i=1}^{t}\log \frac{p_\theta(x_{i-1}\mid x_i)}{q(x_i\mid x_{i-1})}  & \\
&= \log p(x_t) +\left(\sum_{i=1}^{t}\log \frac{p_\theta(x_{i-1}\mid x_i)}{q(x_i\mid x_{i-1}, x_0)} \right) & \text{(Markov assumption)} \\
&= \log p(x_t) +\left(\sum_{i=1}^{t}\log \frac{p_\theta(x_{i-1}\mid x_i)}{q(x_{i-1}\mid x_i, x_0)}\frac{q(x_{i-1}\mid x_0)}{q(x_i\mid x_0)} \right) & \text{(Bayes rule)} \\
&= \log p(x_t) +\left(\sum_{i=1}^{t}\log \frac{p_\theta(x_{i-1}\mid x_i)}{q(x_{i-1}\mid x_i, x_0)}\right)
+ \sum_{i=1}^{t}\log \frac{q(x_{i-1}\mid x_0)}{q(x_i\mid x_0)} & \\
&= \log p(x_t) + \sum_{i=1}^{t}\log \frac{p_\theta(x_{i-1}\mid x_i)}{q(x_{i-1}\mid x_i, x_0)}
+ \sum_{i=1}^{t}\log q(x_{i-1}\mid x_0) - \log q(x_i\mid x_0) & \\
&= \log p(x_t) + \sum_{i=1}^{t}\log \frac{p_\theta(x_{i-1}\mid x_i)}{q(x_{i-1}\mid x_i, x_0)}
- \log q(x_t\mid x_0) & \\
&= \log \frac{p(x_t)}{\log q(x_t\mid x_0)} + \sum_{i=1}^{t}\log \frac{p_\theta(x_{i-1}\mid x_i)}{q(x_{i-1}\mid x_i, x_0)} & \\
&= \log \frac{p(x_t)}{\log q(x_t\mid x_0)} + \sum_{i=2}^{t}\log \frac{p_\theta(x_{i-1}\mid x_i)}{q(x_{i-1}\mid x_i, x_0)} + \log p_\theta(x_0\mid x_1) & \text{(because $q(x_0\mid x_1,x_0) = 1$)}
\end{align*}
```
Plugging this back into the ELBO gives
```math
\begin{align*}
\text{ELBO} &= \mathbb{E}_{z \sim q(\cdot\mid x_0)}
\left[
    \log \frac{p(x_t)}{\log q(x_t\mid x_0)} + \sum_{i=2}^{t}\log \frac{p_\theta(x_{i-1}\mid x_i)}{q(x_{i-1}\mid x_i, x_0)} + \log p_\theta(x_0\mid x_1)
\right] \\
&= \text{KL}(q(x_t\mid x_0), p(x_t)) + \sum_{i=2}^{t}\text{KL}(q(x_{i-1}\mid x_i, x_0), p_\theta(x_{i-1}\mid x_i)) + \log p_\theta(x_0\mid x_1)
\end{align*}
```