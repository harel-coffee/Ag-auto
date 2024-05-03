# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# **April 13. 2024**
#
# \begin{equation}
# \mathbf{y}_{it} = \mathbf{x}^T_{it} \mathbf{\beta} +\varepsilon_{it} + u_{i}
# \end{equation}
#
# The common $u_{i}$ is the usual unit (e.g., country) effect. The correlation across space is implied by the spatial autocorrelation structure
#
# \begin{equation}
# \varepsilon_{it} = \lambda \sum_{j=1}^n W_{ij} \varepsilon_{jt} + v_t
# \end{equation}
#
# ```What is```$v_t$```?```
#
# \begin{equation}
# \mathbf{\varepsilon}_t = (\mathbf{I}_n - \lambda \mathbf{W})^{-1} \mathbf{v}_t,~~~ \mathbf{v}_t =v_t \mathbf{i}
# \end{equation}
#
#
# For when there is spatial autocorrelation in the panel data and when $|\lambda| < 1$ and $\mathbf{I} - \lambda \mathbf{W}$ is non-singular we can write (for $n$ units at time $t$)
#
# \begin{equation}
# \mathbf{y}_t = \mathbf{X}_t \mathbf{\beta} + (\mathbf{I}_n - \lambda \mathbf{W})^{-1} \mathbf{v}_t + \mathbf{u}
# \end{equation}
#
# Assumptions $v_t$ and $u_t$ have mean zero and variances $\sigma^2_v$ and $\sigma_u^2$ and are independent
# across countries and of each other.
#
# The Greene's book: "There is no natural residual based estimator of $\lambda$". It refers to another paper on how to estimate $\lambda$ (Example 11.12). Kelejian and Prucha (1999) have developed a moment-based estimator for $\lambda$ that helps to alleviate the problem (related to the matrix $\mathbf{I}_n - \lambda \mathbf{W}$ and its singularity). Once the estimate of $\lambda$ is in hand, estimation of the spatial autocorrelation model is done by FGLS

# %%
