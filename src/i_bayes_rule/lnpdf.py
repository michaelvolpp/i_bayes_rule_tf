import math

import numpy as np
import tensorflow as tf

from i_bayes_rule.util import cov_to_scale_tril, gmm_log_density_grad_hess, sample_gmm


class LNPDF:
    def log_density(
        self,
        z: tf.Tensor,
        compute_grad: bool = False,
        compute_hess: bool = False,
    ):
        raise NotImplementedError

    def get_num_dimensions(self):
        raise NotImplementedError

    def can_sample(self):
        return False

    def sample(self, n: int):
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

    def log_density(
        self,
        z: tf.Tensor,
        compute_grad: bool = False,
        compute_hess: bool = False,
    ):
        return gmm_log_density_grad_hess(
            z=z,
            log_w=self.log_w,
            loc=self.mu,
            prec=tf.linalg.inv(self.cov),  # TODO: unify prec, scale_tril, cov
            scale_tril=cov_to_scale_tril(self.cov),  # TODO: unify prec, scale_tril, cov
            compute_grad=compute_grad,
            compute_hess=compute_hess,
        )

    def get_num_dimensions(self):
        return self.mu.shape[-1]

    def can_sample(self):
        return True

    def sample(self, n: int):
        return sample_gmm(
            n_samples=tf.constant(n),
            log_w=self.log_w,
            loc=self.mu,
            scale_tril=cov_to_scale_tril(self.cov),
        )


def U(theta: float):
    return np.array(
        [
            [math.cos(theta), math.sin(theta)],
            [-math.sin(theta), math.cos(theta)],
        ]
    )


def make_simple_target(n_tasks: int):
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


def make_star_target(n_tasks: int, n_components: int):
    # Source: Lin et al.

    ## weights
    w_true = np.ones((n_components,)) / n_components

    ## means and precs
    # first component
    mus = [np.array([1.5, 0.0])]
    precs = [np.diag([1.0, 100.0])]
    # other components are generated through rotation
    theta = 2 * math.pi / n_components
    for _ in range(n_components - 1):
        mus.append(U(theta) @ mus[-1])
        precs.append(U(theta) @ precs[-1] @ np.transpose(U(theta)))
    assert len(w_true) == len(mus) == len(precs) == n_components

    mu_true = np.stack(mus, axis=0)
    prec_true = np.stack(precs, axis=0)
    cov_true = np.linalg.inv(prec_true)

    # repeat parameters n_tasks times
    w_true = np.repeat(w_true[None, ...], repeats=n_tasks, axis=0)
    mu_true = np.repeat(mu_true[None, ...], repeats=n_tasks, axis=0)
    cov_true = np.repeat(cov_true[None, ...], repeats=n_tasks, axis=0)

    # check shapes
    assert w_true.shape == (n_tasks, n_components)
    assert mu_true.shape == (n_tasks, n_components, 2)
    assert cov_true.shape == (n_tasks, n_components, 2, 2)

    # generate target dist
    target_dist = GMM_LNPDF(
        target_weights=w_true,
        target_means=mu_true,
        target_covars=cov_true,
    )

    return target_dist
