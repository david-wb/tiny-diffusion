# Tiny Diffusion

A minimal implementation of a denoising diffusion model for image generation.

# Diffusion Process

The "diffusion" process is a Markov chain that adds Gaussian noise to a given image $x_0$ at each step. The posterior distribution of the
diffused image samples are given by

```math
q(x_1, \ldots, x_t \mid  x_0) = \prod_{i=1}^{t}q(x_i \mid  x_{i-1}).
```

where transition probabilities are isotropic Gaussians of the form

```math
q(x_t \mid  x_{t-1}) = \mathcal{N}(x_{t}; \sqrt{\alpha_{t}} x_{t-1}, (1 - \alpha_t) I).
```

We can equivalently write the state transitions as a scaled value of the previous state added to a random noise term:

```math
x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1 - \alpha_t}\epsilon_{t-1}
```

where each $\epsilon_t \sim \mathcal{N}(\cdot; 0,  I)$ is unit noise.

An important property of the diffusion process is that it is possible to express the distribution $q(x_t | x_0)$ in closed form
without, which means we can directly sample $x_t$ given $x_0$ without going through all of the intermediate states. To see this, first note that

```math
\begin{align*}
x_t &= \sqrt{\alpha_t}x_{t-1} + \sqrt{1 - \alpha_t}\epsilon_t \\
 &= \sqrt{\alpha_t}\sqrt{\alpha_{t-1}}x_{t-2} + \sqrt{\alpha_t}\sqrt{1 - \alpha_{t-1}}\epsilon_{t-2} + \sqrt{1-\alpha_t}\epsilon_{t-1}.
\end{align*}
```

The because $\epsilon_{t-2}$ and $\epsilon_{t-1}$ are independent and identically distributed random variables, the last two terms can be combined into a single distribution

```math
\begin{align*}
\sqrt{\alpha_t}\sqrt{1 - \alpha_{t-1}}\epsilon_{t-2} + \sqrt{1-\alpha_t}\epsilon_{t-1}
&\sim \mathcal{N}(0, \alpha_t(1-\alpha_{t-1})I) + \mathcal{N}(0, (1-\alpha_t)I) \\
&\sim \mathcal{N}(0, \alpha_t(1-\alpha_{t-1})I + (1-\alpha_t)I) \\
&\sim \mathcal{N}(0, (1 - \alpha_t\alpha_{t-1})I) \\
&\sim \sqrt{1 - \alpha_t\alpha_{t-1}}\epsilon_{t-2}
\end{align*}
```

Thus, the marginal distribution of $x_t$ given $x_0$ can be expressed in closed form as a function of $x_0$ and a unit-noise variable $\epsilon_0$.

```math
\begin{align*}
x_t &= \sqrt{\alpha_t}x_{t-1} + \sqrt{1 - \alpha_t}\epsilon_t \\
 &= \sqrt{\alpha_t\alpha_{t-1}}x_{t-2} + \sqrt{1 - \alpha_t\alpha_{t-1}}\epsilon_{t-2} \\
 &= \ldots \\
 &= \sqrt{\bar{\alpha}_t}x_{0} + \sqrt{1 - \bar{\alpha}_t}\epsilon_{0} \\
&= \sqrt{\bar{\alpha}_t}x_{0} + \sqrt{1 - \bar{\alpha}_t}\epsilon
\end{align*}
```

for some unit noise $\epsilon$ and where

```math
    \bar{\alpha}_t = \prod_{i=1}^{t}\alpha_i.
```

Thus, the probability of any given state $x_t$ given a starting state $x_0$ is

```math
    q(x_t \mid  x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_{0}, (1 - \bar{\alpha}_t) I)
```

Also, by rearranging, we can equivalently write $x_0$ in terms of $x_t$ and a noise term
```
x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1 - \bar{\alpha}_t}\epsilon)
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

The transitions are learned Gaussians with mean and variance parameterized by $\theta$:

```math
p_\theta(x_{i-1}\mid x_i) = \mathcal{N}(x_{i-1}; \mu_{\theta}(x_i, i), \Sigma_{\theta}(x_i, i)).
```

The functions $\mu_\theta(x_i, i)$ and $\Sigma_{\theta}(x_i, i)$ represent a parametric model, typically a neural network, which takes as inputs an image $x_i$ and the associated timestep $i$, and outputs the mean and variance of a Gaussian distribution for the denoised sample $x_{i-1}$. The objective is to learn the parameters $\theta$ of this model.

## Variational Inference for MLE

As we will see, we can learn the reverse process parameters $\theta$ by maximising the likelihood of the data

```math
\mathbb{E}_{x_0}[p_\theta(x_0)] = \mathbb{E}_{x_0}\left[\int p_\theta(x_0, x_1, \ldots, x_t) d\mathbf{x}_{1:t}\right]
```

Our training objective is thus to find the optional parameters $\theta^{*}$:

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

where

```math
\begin{align*}
\text{ELBO} &= \mathbb{E}_{z \sim q(\cdot\mid x_0)}\left[\log \frac{p_\theta(x_0, z)}{q(z\mid x_0)}\right].
\end{align*}
```

Because the KL term is positive, maximizing ELBO increases the data likelihood _and_ makes $p_\theta(z \mid  x_0)$ closer to the true posterior $q(z\mid x_0)$.

Expanding the term inside the ELBO expectation we get

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
&= \log \frac{p(x_t)}{q(x_t\mid x_0)} + \sum_{i=1}^{t}\log \frac{p_\theta(x_{i-1}\mid x_i)}{q(x_{i-1}\mid x_i, x_0)} & \\
&= \log \frac{p(x_t)}{q(x_t\mid x_0)} + \sum_{i=2}^{t}\log \frac{p_\theta(x_{i-1}\mid x_i)}{q(x_{i-1}\mid x_i, x_0)} + \log p_\theta(x_0\mid x_1) & q(x_0\mid x_1,x_0) = 1
\end{align*}
```

Plugging this back into the ELBO gives

```math
\begin{align*}
\text{ELBO} &= \mathbb{E}_{z \sim q(\cdot\mid x_0)}
\left[
    \log \frac{p(x_t)}{q(x_t\mid x_0)} + \sum_{i=2}^{t}\log \frac{p_\theta(x_{i-1}\mid x_i)}{q(x_{i-1}\mid x_i, x_0)} + \log p_\theta(x_0\mid x_1)
\right] \\
&= \text{KL}(q(x_t\mid x_0), p(x_t)) + \sum_{i=2}^{t}\text{KL}(q(x_{i-1}\mid x_i, x_0), p_\theta(x_{i-1}\mid x_i)) + \mathbb{E}_{z \sim q(\cdot\mid x_0)}[\log p_\theta(x_0\mid x_1)]
\end{align*}
```

Note that the first term is constant with respect to $\theta$ and can therefore be ignored for the purpose of optimization.

For the terms inside the summation, it's clear that we want to make $p_\theta(x_{i-1}\mid x_i)$ as close as possible to $q(x_{i-1}\mid x_i, x_0)$. We therefore need derive the distribution $q(x_{i-1}\mid x_i, x_0)$, which is a backward conditional probability of the forward process Markov chain. We can then use the mean and variance of this distribution as training targets for our model $p_\theta(x_{i-1}\mid x_i)$.

## Deriving Forward Process Backward Conditional Probabilities

Here is the strategy. First we will derive the joint Gaussian distribution $q(x_i, x_{i-1}\mid x_0)$. Then we will use the conditional Gaussian formula to derive the mean and variance of $q(x_i \mid x_{i-1}, x_0)$, which is also Gaussian.

We know that for two multi-variate Gaussian random variables $\mathbf{X}$ and $\mathbf{Y}$, their joint distribution is given by

```math
\begin{bmatrix}
\mathbf{X} \\
\mathbf{Y}
\end{bmatrix}
\sim \mathcal{N} \left(
\begin{bmatrix}
\boldsymbol{\mu}_X \\
\boldsymbol{\mu}_Y
\end{bmatrix},
\begin{bmatrix}
\boldsymbol{\Sigma}_X & \boldsymbol{\Sigma}_{XY} \\
\boldsymbol{\Sigma}_{XY}^\top & \boldsymbol{\Sigma}_Y
\end{bmatrix}
\right)
```

Thus we can substitute $\mathbf{X} \sim q(x_i \mid x_0)$ and $\mathbf{Y} \sim q(x_{i-1} \mid x_0)$ and get

```math
q(x_i, x_{i-1} \mid x_0) = \mathcal{N} \left(
\begin{bmatrix}
\sqrt{\bar{\alpha}_{i}}x_{0} \\
\sqrt{\bar{\alpha}_{i-1}}x_{0}
\end{bmatrix},
\begin{bmatrix}
(1 - \bar{\alpha}_{i})I & \boldsymbol{\Sigma}_{XY} \\
\boldsymbol{\Sigma}_{XY}^\top & (1 - \bar{\alpha}_{i-1})I
\end{bmatrix}
\right)
```

Now we just need to find $\boldsymbol{\Sigma}_{XY}$ which we can get from the definition of covariance. For brevity I will drop the notation $\mid x_0$, but remember we are conditioning on $x_0$.

First, let's use the expressions for $x_i$ and $x_{i-1}$ given $x_0$ in terms of random noise variables $\epsilon_t$.

```math
x_{i-1} = \sqrt{\bar{\alpha}_{i-1}}x_{0} + \sqrt{1 - \bar{\alpha}_{i-1}}\epsilon_{0}
```

For $x_i$, remember that is is dependent $x_{i-1}$, so we must use the single-step expression:

```math
\begin{align*}
x_{i} &= \sqrt{\alpha_i}x_{i-1} + \sqrt{1 - \alpha_i}\epsilon_{i-1} \\
&= \sqrt{\alpha_i}(\sqrt{\bar{\alpha}_{i-1}}x_{0} + \sqrt{1 - \bar{\alpha}_{i-1}}\epsilon_{0}) + \sqrt{1 - \alpha_i}\epsilon_{i-1} \\
&= \sqrt{\bar{\alpha}_{i}}x_{0} + \sqrt{\alpha_i (1 - \bar{\alpha}_{i-1})}\epsilon_{0} + \sqrt{1 - \alpha_i}\epsilon_{i-1}
\end{align*}
```

Okay, so now we have $x_i$ and $x_{i-1}$ expressed in terms of two **independent** random noise variables $\epsilon_{i-1}$ and $\epsilon_0$. Plugging these into the definition for covariance, we get

```math
\begin{align*}
\boldsymbol{\Sigma}_{XY} &= \mathbb{E}_q\left[(x_i - \mathbb{E}_q[x_i])(x_{i-1} - \mathbb{E}_q[x_{i-1}])\right] \\
&= \mathbb{E}_q
\left[
    (x_i - \sqrt{\bar{\alpha}_{i}}x_0)(x_{i-1} - \sqrt{\bar{\alpha}_{i-1}}x_0)
\right] \\
&= \mathbb{E}_q
\left[
    (\sqrt{\bar{\alpha}_{i}}x_{0} + \sqrt{\alpha_i (1 - \bar{\alpha}_{i-1})}\epsilon_{0} + \sqrt{1 - \alpha_i}\epsilon_{i-1} - \sqrt{\bar{\alpha}_{i}}x_0)
    (\sqrt{\bar{\alpha}_{i-1}}x_{0} + \sqrt{1 - \bar{\alpha}_{i-1}}\epsilon_{0}  - \sqrt{\bar{\alpha}_{i-1}}x_0)
\right] \\
&= \mathbb{E}_q
\left[
    (\sqrt{\alpha_i (1 - \bar{\alpha}_{i-1})}\epsilon_{0} + \sqrt{1 - \alpha_i}\epsilon_{i-1})
    (\sqrt{1 - \bar{\alpha}_{i-1}}\epsilon_{0})
\right] \\
&= \mathbb{E}_q
\left[
    \sqrt{\alpha_i}(1 - \bar{\alpha}_{i-1})\epsilon_{0}^2 + \sqrt{1 - \alpha_i}\sqrt{1 - \bar{\alpha}_{i-1}}\epsilon_{i-1}\epsilon_{0}
\right] \\
&= \sqrt{\alpha_i}(1 - \bar{\alpha}_{i-1})I  + 0 \\
&= \sqrt{\alpha_i}(1 - \bar{\alpha}_{i-1})I
\end{align*}
```

And now finally we have the following joint distribution for $x_i$ and $x_{i-1}$ conditioned on $x_0$:

```math
q(x_i, x_{i-1} \mid x_0) = \mathcal{N} \left(
\begin{bmatrix}
\sqrt{\bar{\alpha}_{i}}x_{0} \\
\sqrt{\bar{\alpha}_{i-1}}x_{0}
\end{bmatrix},
\begin{bmatrix}
(1 - \bar{\alpha}_{i})I & \sqrt{\alpha_i}(1 - \bar{\alpha}_{i-1})I \\
\sqrt{\alpha_i}(1 - \bar{\alpha}_{i-1})I & (1 - \bar{\alpha}_{i-1})I
\end{bmatrix}
\right)
```

### Conditional Gaussian Formula

Now that we have a joint normal distribution for $x_{i-1}$ and $x_{i}$, we can use the [conditional Gaussian formula](https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions) to derive $q(x_{i-1} \mid x_i, x_0)$. Note that although we are conditioning on $x_0$, the joint distribution $q(x_{i-1}, x_i \mid x_0)$ remains a valid probability distribution, so the conditional Gaussian formula still applies to $x_{i-1}$ and $x_{i}$ while treating $x_0$ like a constant.

Substituting $X = x_i$ and $Y = x_{i-1}$, the formula tells us that

```math
\begin{align*}
&= \sqrt{\bar{\alpha}_{i-1}}x_{0} + \frac{\sqrt{\alpha_i}(1 - \bar{\alpha}_{i-1})}{(1 - \bar{\alpha}_{i})}(x_{i} - \sqrt{\bar{\alpha}_{i}}x_{0})  \\
&= \left(\sqrt{\bar{\alpha}_{i-1}} - \sqrt{\bar{\alpha}_{i}}\sqrt{\alpha_i}\frac{(1 - \bar{\alpha}_{i-1})}{(1 - \bar{\alpha}_{i})}\right)x_0
+ \sqrt{\alpha_i}\frac{(1 - \bar{\alpha}_{i-1})}{(1 - \bar{\alpha}_{i})}x_{i} \\
&= \sqrt{\bar{\alpha}_{i-1}}\left(1 - \alpha_{i}\frac{(1 - \bar{\alpha}_{i-1})}{(1 - \bar{\alpha}_{i})}\right)x_0
+ \sqrt{\alpha_i}\frac{(1 - \bar{\alpha}_{i-1})}{(1 - \bar{\alpha}_{i})}x_{i} \\
&= \sqrt{\bar{\alpha}_{i-1}}\left(\frac{1 - \bar{\alpha}_{i} - \alpha_i +\bar{\alpha}_{i}}{1 - \bar{\alpha}_{i}}\right)x_0
+ \sqrt{\alpha_i}\frac{(1 - \bar{\alpha}_{i-1})}{(1 - \bar{\alpha}_{i})}x_{i} \\
&= \boxed{\frac{\sqrt{\bar{\alpha}_{i-1}}}{1 - \bar{\alpha}_{i}}(1 - \alpha_i)x_0
+ \sqrt{\alpha_i}\frac{(1 - \bar{\alpha}_{i-1})}{(1 - \bar{\alpha}_{i})}x_{i}}
\end{align*}
```

and for the variance

```math
\begin{align*}
\boldsymbol{\Sigma}_{Y\mid X} &= \boldsymbol{\Sigma}_{Y} -\boldsymbol{\Sigma}_{YX} \boldsymbol{\Sigma}_{X}^{-1}\boldsymbol{\Sigma}_{XY}  \\
&= (1 - \bar{\alpha}_{i-1})I - \sqrt{\alpha_i}(1 - \bar{\alpha}_{i-1})\frac{I}{(1 - \bar{\alpha}_{i})}\sqrt{\alpha_i}(1 - \bar{\alpha}_{i-1}) \\
&= (1 - \bar{\alpha}_{i-1})I - \frac{\alpha_i(1 - \bar{\alpha}_{i-1})^2}{(1 - \bar{\alpha}_{i})}I \\
&= (1 - \bar{\alpha}_{i-1})\left(1 - \frac{\alpha_i(1 - \bar{\alpha}_{i-1})}{1 - \bar{\alpha}_i}\right)I \\
&= (1 - \bar{\alpha}_{i-1})\left(\frac{1 - \bar{\alpha}_i - \alpha_i + \bar{\alpha}_{i}}{1 - \bar{\alpha}_i}\right)I \\
&= \boxed{\frac{1 - \bar{\alpha}_{i-1}}{1 - \bar{\alpha}_i}\left(1 - \alpha_i\right)I}
\end{align*}
```

## Training the Denoising Model

Alright, so now we have targets for the mean and variance of our probabilistic denoising model $p_\theta(x_{i-1} | x_i)$, which we want to make as close as possible to $q(x_{i-1}
\mid x_i, x_0)$.

The KL divergence of two isotropic normal distributions, $p$ and $q$, with variances $\sigma_1^2$ and $\sigma_2^2$, respectively is given by 


```math
\begin{align*}
\text{KL}(p, q) = \frac{1}{2} \left[ \log \frac{\sigma_2^2}{\sigma_1^2} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{\sigma_2^2} - 1 \right]
\end{align*}
```

Let's define our target mean and variance as 

```math
\tilde{\mu}(x_i, x_0) = \frac{\sqrt{\bar{\alpha}_{i-1}}}{1 - \bar{\alpha}_{i}}(1 - \alpha_i)x_0
+ \sqrt{\alpha_i}\frac{(1 - \bar{\alpha}_{i-1})}{(1 - \bar{\alpha}_{i})}x_{i}
```

```math
\tilde{\sigma}^2(i) = \frac{1 - \bar{\alpha}_{i-1}}{1 - \bar{\alpha}_i}\left(1 - \alpha_i\right)
```

If we define our model such that

```math
p_\theta(x_{i-1}\mid x_i) = \mathcal{N}(x_{t}; \mu_\theta(x_{i}, i)), \tilde{\sigma}^2(i)I)
```

then our objective becomes

```math
\begin{align*}
\text{KL}(q(x_{i-1}\mid x_i, x_0), p_\theta(x_{i-1}\mid x_i)) &= \frac{1}{2} \left[ \log \frac{\tilde{\sigma}^2(i)}{\tilde{\sigma}^2(i)} + \frac{\tilde{\sigma}^2(i) + (\tilde{\mu}(x_{i}, x_0)- \mu_\theta(x_{i}, i)))^2}{\tilde{\sigma}^2(i)} - 1 \right] \\
&= \frac{1}{2}\frac{(\tilde{\mu}(x_{i}, x_0)- \mu_\theta(x_{i}, i)))^2}{\tilde{\sigma}^2(i)} + C \\
\end{align*}
```
Where $C$ is a constant with respect to $\theta$.

If we expand the target mean as using the closed form expression for $x_0$ in terms of $x_i$ and a random noise term, we get

```math
\begin{align*}
\tilde{\mu}(x_i, x_0) &= \frac{\sqrt{\bar{\alpha}_{i-1}}}{1 - \bar{\alpha}_{i}}(1 - \alpha_i)x_0
+ \sqrt{\alpha_i}\frac{(1 - \bar{\alpha}_{i-1})}{(1 - \bar{\alpha}_{i})}x_{i} \\
&= \frac{\sqrt{\bar{\alpha}_{i-1}}}{1 - \bar{\alpha}_{i}}(1 - \alpha_i)\left(\frac{1}{\sqrt{\bar{\alpha}_i}}(x_i - \sqrt{1 - \bar{\alpha}_i}\epsilon)\right)
+ \sqrt{\alpha_i}\frac{(1 - \bar{\alpha}_{i-1})}{(1 - \bar{\alpha}_{i})}x_i \\
&= \frac{1}{\sqrt{\alpha_i}}\frac{1 - \alpha_i}{1 - \bar{\alpha}_{i}}\left(x_i - \sqrt{1 - \bar{\alpha}_i}\epsilon\right)
+ \sqrt{\alpha_i}\frac{(1 - \bar{\alpha}_{i-1})}{(1 - \bar{\alpha}_{i})}x_i \\

&= \frac{1 - \alpha_i}{\sqrt{\alpha_i}\sqrt{1 - \bar{\alpha}_{i}}}\epsilon
+ \frac{1}{\sqrt{\alpha_i}}\left(\frac{1 - \alpha_i}{1 - \bar{\alpha}_{i}} + \alpha_i\frac{(1 - \bar{\alpha}_{i-1})}{(1 - \bar{\alpha}_{i})}\right)x_i \\

&= \frac{1 - \alpha_i}{\sqrt{\alpha_i}\sqrt{1 - \bar{\alpha}_{i}}}\epsilon
+ \frac{1}{\sqrt{\alpha_i}}\left(\frac{1 - \alpha_i + \alpha_i - \bar{\alpha}_i}{1 - \bar{\alpha}_{i}}\right)x_i \\
&= \frac{1 - \alpha_i}{\sqrt{\alpha_i}\sqrt{1 - \bar{\alpha}_{i}}}\epsilon
+ \frac{1}{\sqrt{\alpha_i}}x_i \\
\end{align*}
```
This means we can reparameterize the mean of our denoising model $\mu_\theta$ as a function of $x_i$ and a learned noise term, and instead of predicting the mean directly, we can predict the noise instead.

## Denoising Procedure
In order to "denoise" an image $x_i$ by one step, i.e. draw a sample from $p_\theta(x_{i-1}\mid x_i)$, we first sample

```math
\epsilon \sim \epsilon_\theta(x_i, i)
```
and
```math
z \sim \mathcal{N}(\cdot; 0, \tilde{\sigma}^2(i)I)
```
and then compute
```math
x_{i-1} \coloneqq \frac{1 - \alpha_i}{\sqrt{\alpha_i}\sqrt{1 - \bar{\alpha}_{i}}}\epsilon
+ \frac{1}{\sqrt{\alpha_i}}x_i + z
```
and repeat for $x_{i-2}$ and so on down to $x_0$.

## Training Procedure

The last part is to learn the model for $\epsilon_\theta(x_i, i)$. We can do this as follows.

First select an image $x_0$ and time index $i$ at random 
```math
x_0 \sim \Omega_{\text{images}}
```
```math
i \sim [1, T]
```
Then draw a random unit noise variable
```math
\epsilon \sim \mathcal{N}(\cdot; 1, I)
```
and compute
```math
x_i := \sqrt{\bar{\alpha}_i}x_{0} + \sqrt{1 - \bar{\alpha}_i}\epsilon
```
This gives us our "diffused" sample $x_i$ and the total noise that produced it from $x_0$.

Finally, all we have to do is minimize the loss

```math
L(\theta) = \|\epsilon_\theta(x_i, i) - \epsilon\|^2
```
via gradient descent.