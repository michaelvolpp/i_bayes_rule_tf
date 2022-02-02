import math

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

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
    def __init__(self, target_weights, target_means, target_covars):
        self.target_weights = target_weights
        self.target_means = target_means
        self.target_covars = target_covars
        self.target_means = target_means.astype(np.float32)
        self.target_covars = target_covars.astype(np.float32)
        self.gmm = tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(
                logits=np.log(target_weights).astype(np.float32)
            ),
            components_distribution=tfp.distributions.MultivariateNormalTriL(
                loc=target_means.astype(np.float32),
                scale_tril=np.linalg.cholesky(target_covars).astype(np.float32),
            ),
        )

    def log_density(self, x):
        x = tf.cast(x, dtype=tf.float32)
        return self.gmm.log_prob(x)

    def get_num_dimensions(self):
        return len(self.target_means[0])

    def can_sample(self):
        return True

    def sample(self, n):
        return self.gmm.sample(n)


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


def make_simple_target():
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
