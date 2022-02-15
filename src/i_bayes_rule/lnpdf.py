import math

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from i_bayes_rule.util import (
    sample_gmm,
    gmm_log_density,
    cov_to_scale_tril,
    gmm_log_component_densities,
)

# Source: Oleg


class LNPDF:
    def log_density(self, x):
        raise NotImplementedError

    def gradient_log_density(self, x):
        raise NotImplementedError

    def get_num_dimensions(self):
        raise NotImplementedError

    def can_sample(self):
        return False

    def sample(self, n):
        raise NotImplementedError


class GMM_LNPDF(LNPDF):
    def __init__(
        self,
        target_weights: np.ndarray,
        target_means: np.ndarray,
        target_covars: np.ndarray,
    ):
        self.log_w = tf.math.log(tf.constant(target_weights.astype(np.float32)))
        self.mu = tf.constant(target_means.astype(np.float32))
        self.cov = tf.constant(target_covars.astype(np.float32))

    def get_num_dimensions(self):
        return self.mu.shape[-1]

    def can_sample(self):
        return True

    @tf.function
    def sample(self, n):
        return sample_gmm(
            n_samples=n,
            log_w=self.log_w,
            loc=self.mu,
            scale_tril=cov_to_scale_tril(self.cov),
        )

    @tf.function
    def log_density(self, x):
        return gmm_log_density(
            z=x,
            log_w=self.log_w,
            log_component_densities=gmm_log_component_densities(
                z=x, loc=self.mu, scale_tril=cov_to_scale_tril(self.cov)
            ),
        )


def make_target(num_dimensions):
    num_true_components = 10
    weights = np.ones(num_true_components) / num_true_components
    means = np.empty((num_true_components, num_dimensions))
    covs = np.empty((num_true_components, num_dimensions, num_dimensions))
    for i in range(0, num_true_components):
        means[i] = 100 * (np.random.random(num_dimensions) - 0.5)
        covs[i] = 0.1 * np.random.normal(
            0, num_dimensions, (num_dimensions * num_dimensions)
        ).reshape((num_dimensions, num_dimensions))
        covs[i] = covs[i].transpose().dot(covs[i])
        covs[i] += 1 * np.eye(num_dimensions)
    return GMM_LNPDF(weights, means, covs)


def U(theta):
    return np.array(
        [
            [math.cos(theta), math.sin(theta)],
            [-math.sin(theta), math.cos(theta)],
        ]
    )


def make_simple_target(n_tasks):
    pi = math.pi

    # weights
    w_true = np.array([0.5, 0.3, 0.2])

    # means
    mu_true = np.array(
        [
            [-2.0, -2.0],
            [2.0, -2.0],
            [0.0, 2.0],
        ]
    )

    # covs
    cov1 = np.array([[0.5, 0.0], [0.0, 1.0]])
    cov1 = U(pi / 4) @ cov1 @ np.transpose(U(pi / 4))
    cov2 = np.array([[0.5, 0.0], [0.0, 1.0]])
    cov2 = U(-pi / 4) @ cov2 @ np.transpose(U(-pi / 4))
    cov3 = np.array([[1.0, 0.0], [0.0, 2.0]])
    cov3 = U(pi / 2) @ cov3 @ np.transpose(U(pi / 2))
    cov_true = np.stack([cov1, cov2, cov3], axis=0)

    # stack the parameters n_tasks times
    w_true = np.stack([w_true] * n_tasks, axis=0)
    mu_true = np.stack([mu_true] * n_tasks, axis=0)
    cov_true = np.stack([cov_true] * n_tasks, axis=0)

    # generate target dist
    target_dist = GMM_LNPDF(
        target_weights=w_true,
        target_means=mu_true,
        target_covars=cov_true,
    )

    return target_dist


def make_star_target(num_components):
    # Source: Lin et al.
    K = num_components

    ## weights
    w_true = np.ones((K,)) / K

    ## means and precs
    # first component
    mus = [np.array([1.5, 0.0])]
    precs = [np.diag([1.0, 100.0])]
    # other components are generated through rotation
    theta = 2 * math.pi / K
    for _ in range(K - 1):
        mus.append(U(theta) @ mus[-1])
        precs.append(U(theta) @ precs[-1] @ np.transpose(U(theta)))
    assert len(w_true) == len(mus) == len(precs) == K

    mu_true = np.stack(mus, axis=0)
    prec_true = np.stack(precs, axis=0)
    cov_true = np.linalg.inv(prec_true)

    # generate target dist
    target_dist = GMM_LNPDF(
        target_weights=w_true,
        target_means=mu_true,
        target_covars=cov_true,
    )

    return target_dist
