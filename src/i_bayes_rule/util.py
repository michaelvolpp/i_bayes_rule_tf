import math

import tensorflow as tf
import tensorflow_probability as tfp
import itertools


@tf.function
def prec_to_prec_tril(prec: tf.Tensor):
    # Compute lower cholesky factors of precision matrices

    # check input
    d_z = prec.shape[-1]
    assert prec.shape[-2:] == (d_z, d_z)

    # from: https://pytorch.org/docs/stable/_modules/torch/distributions/multivariate_normal.html#MultivariateNormal
    prec_tril = tf.linalg.cholesky(prec)

    # check output
    assert prec_tril.shape[-2:] == (d_z, d_z)
    return prec_tril


@tf.function
def prec_to_scale_tril(prec: tf.Tensor):
    # Compute lower cholesky factors of covariance matrices from precision matrices

    # check input
    d_z = prec.shape[-1]
    assert prec.shape[-2:] == (d_z, d_z)

    # from: https://pytorch.org/docs/stable/_modules/torch/distributions/multivariate_normal.html#MultivariateNormal
    Lf = tf.linalg.cholesky(tf.reverse(prec, axis=(-2, -1)))
    L_inv = tf.linalg.matrix_transpose(tf.reverse(Lf, axis=(-2, -1)))
    scale_tril = tf.linalg.triangular_solve(
        L_inv, tf.eye(d_z, dtype=L_inv.dtype), lower=True
    )

    # check output
    assert scale_tril.shape[-2:] == (d_z, d_z)
    return scale_tril


@tf.function
def scale_tril_to_cov(scale_tril: tf.Tensor):
    # check input
    d_z = scale_tril.shape[-1]
    assert scale_tril.shape[-2:] == (d_z, d_z)

    # compute cov from scale_tril
    cov = tf.matmul(scale_tril, tf.linalg.matrix_transpose(scale_tril))

    # check output
    assert cov.shape[-2:] == (d_z, d_z)
    return cov


@tf.function
def cov_to_scale_tril(cov: tf.Tensor):
    # check input
    d_z = cov.shape[-1]
    assert cov.shape[-2:] == (d_z, d_z)

    # compute scale_tril from cov
    scale_tril = tf.linalg.cholesky(cov)

    # check output
    assert scale_tril.shape[-2:] == (d_z, d_z)
    return scale_tril


# def prec_to_cov(prec: tf.Tensor):
#     # check input
#     #assert tf.rank(prec) == 3
#     K = prec.shape[0]
#     D = prec.shape[1]
#     #assert prec.shape[2] == D

#     # compute cov from prec
#     # TODO: is there a better way? use prec_to_scale_tril?
#     cov = tf.linalg.inv(prec)

#     # check output
#     #assert cov.shape == (K, D, D)
#     return prec


# def cov_to_prec(cov: tf.Tensor):
#     # check input
#     #assert tf.rank(cov) == 3
#     K = cov.shape[0]
#     D = cov.shape[1]
#     #assert cov.shape[2] == D

#     # compute cov from prec
#     # TODO: is there a better way? use prec_to_scale_tril?
#     prec = tf.linalg.inv(cov)

#     # check output
#     #assert prec.shape == (K, D, D)
#     return prec


# @tf.function
# def sample_categorical(n_samples: int, log_w: tf.Tensor):
#     # check input
#     batch_shape = log_w.shape[:-1]

#     # sample categorical
#     thresholds = tf.cumsum(tf.exp(log_w), axis=-1)
#     eps = tf.random.uniform(shape=(n_samples,) + batch_shape)
#     samples = tf.argmax(
#         eps[..., None] < thresholds[None, ...],
#         axis=-1,
#         output_type=tf.int32,
#     )

#     # check output
#     assert samples.shape == (n_samples,) + batch_shape
#     return samples


# @tf.function
# def sample_gaussian(n_samples: int, loc: tf.Tensor, scale_tril: tf.Tensor):
#     # check input
#     batch_shape = loc.shape[:-1]
#     d_z = loc.shape[-1]
#     assert loc.shape == batch_shape + (d_z,)
#     assert scale_tril.shape == batch_shape + (d_z, d_z)

#     # sample
#     samples = tf.einsum(
#         "...ij,s...j->s...i",
#         scale_tril,
#         tf.random.normal((n_samples,) + batch_shape + (d_z,), mean=0.0, stddev=1.0),
#     )
#     samples = samples + loc[None, ...]

#     # check output
#     assert samples.shape == (n_samples,) + batch_shape + (d_z,)
#     return samples


@tf.function
def sample_gmm(n_samples: int, log_w: tf.Tensor, loc: tf.Tensor, scale_tril: tf.Tensor):
    # check input
    d_z = loc.shape[-1]
    n_components = loc.shape[-2]
    batch_shape = loc.shape[:-2]
    assert log_w.shape == batch_shape + (n_components,)
    assert loc.shape == batch_shape + (n_components, d_z)
    assert scale_tril.shape == batch_shape + (n_components, d_z, d_z)

    # sample gmm
    samples = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(
            logits=log_w, validate_args=False
        ),
        components_distribution=tfp.distributions.MultivariateNormalTriL(
            loc=loc, scale_tril=scale_tril, validate_args=False
        ),
        validate_args=False,
    ).sample(n_samples)

    # TODO: write this without using tfp.distributions?
    # non-batched code
    # sampled_components = sample_categorical(n_samples=n_samples, log_w=log_w)
    # samples = tf.zeros((n_samples,) + batch_shape + (d_z,))
    # for i in range(n_components):
    #     n_samples = tf.reduce_sum(tf.cast(sampled_components == i, tf.int32))
    #     cur_samples = sample_gaussian(d_z, loc[i], scale_tril[i], n_samples)
    #     samples.append(cur_samples)
    # samples = tf.random.shuffle(tf.concat(samples, axis=0))

    # check output
    assert samples.shape == (n_samples,) + batch_shape + (d_z,)
    return samples


@tf.function
def gmm_log_component_densities(z: tf.Tensor, loc: tf.Tensor, scale_tril: tf.Tensor):
    # check input
    n_samples = z.shape[0]
    batch_shape = z.shape[1:-1]
    n_components = loc.shape[-2]
    d_z = z.shape[-1]
    assert z.shape == (n_samples,) + batch_shape + (d_z,)
    assert loc.shape == batch_shape + (n_components, d_z)
    assert scale_tril.shape == batch_shape + (n_components, d_z, d_z)

    # compute log component densities
    diffs = z[..., None, :] - loc[None, ...]
    assert diffs.shape == (n_samples,) + batch_shape + (n_components, d_z)
    # TODO: can we get away w/o broadcast_to by using sample-dim as last dim of diffs?
    sqrts = tf.linalg.triangular_solve(
        tf.broadcast_to(scale_tril[None, ...], diffs.shape + (d_z,)), diffs[..., None]
    )
    sqrts = tf.squeeze(sqrts, axis=-1)
    assert sqrts.shape == (n_samples,) + batch_shape + (n_components, d_z)
    mahalas = -0.5 * tf.reduce_sum(sqrts * sqrts, axis=-1)
    assert mahalas.shape == (n_samples,) + batch_shape + (n_components,)
    const_parts = -0.5 * tf.reduce_sum(
        tf.math.log(tf.square(tf.linalg.diag_part(scale_tril))), axis=-1
    ) - 0.5 * d_z * tf.math.log(2 * math.pi)
    assert const_parts.shape == batch_shape + (n_components,)
    log_component_densities = mahalas + const_parts[None, ...]

    # check output
    assert log_component_densities.shape == (n_samples,) + batch_shape + (n_components,)
    return log_component_densities


@tf.function
def gmm_log_density(
    z: tf.Tensor,
    log_w: tf.Tensor,
    log_component_densities: tf.Tensor,
):
    # check input
    n_samples = z.shape[0]
    d_z = z.shape[-1]
    n_components = log_w.shape[-1]
    batch_shape = log_w.shape[:-1]
    assert z.shape == (n_samples,) + batch_shape + (d_z,)
    assert log_w.shape == batch_shape + (n_components,)
    assert log_component_densities.shape == (n_samples,) + batch_shape + (n_components,)

    # compute log density
    log_joint_densities = log_component_densities + log_w[None, ...]
    assert log_joint_densities.shape == (n_samples,) + batch_shape + (n_components,)
    log_density = tf.reduce_logsumexp(log_joint_densities, axis=-1)

    # check output
    assert log_density.shape == (n_samples,) + batch_shape
    return log_density


@tf.function
def gmm_log_responsibilities(
    z: tf.Tensor,
    log_w: tf.Tensor,
    log_component_densities: tf.Tensor,
    log_density: tf.Tensor,
):
    # check input
    n_samples = z.shape[0]
    d_z = z.shape[-1]
    n_components = log_w.shape[-1]
    batch_shape = log_w.shape[:-1]
    assert z.shape == (n_samples,) + batch_shape + (d_z,)
    assert log_w.shape == batch_shape + (n_components,)
    assert log_density.shape == (n_samples,) + batch_shape
    assert log_component_densities.shape == (n_samples,) + batch_shape + (n_components,)

    # compute log density and responsibilities
    log_joint_densities = log_component_densities + log_w[None, ...]
    assert log_joint_densities.shape == (n_samples,) + batch_shape + (n_components,)
    log_responsibilities = log_joint_densities - log_density[..., None]

    # check output
    assert log_responsibilities.shape == (n_samples,) + batch_shape + (n_components,)
    return log_responsibilities


def gmm_log_density_grad_hess(
    z: tf.Tensor,
    log_w: tf.Tensor,
    loc: tf.Tensor,
    prec: tf.Tensor,  # TODO: unify usage of prec, cov, scale_tril
    scale_tril: tf.Tensor,  # TODO: unify usage of prec, cov, scale_tril
    compute_gradient: bool,
    compute_hessian: bool,
):
    # check input
    n_samples = z.shape[0]
    d_z = loc.shape[-1]
    n_components = loc.shape[-2]
    batch_shape = loc.shape[:-2]
    assert z.shape == (n_samples,) + batch_shape + (d_z,)
    assert log_w.shape == batch_shape + (n_components,)
    assert loc.shape == batch_shape + (n_components, d_z)
    assert prec.shape == batch_shape + (n_components, d_z, d_z)
    assert not (compute_hessian and not compute_gradient)

    # compute log_density, and log_responsibilities
    log_density, log_component_densities = _gmm_log_density_and_log_component_densities(
        z=z, log_w=log_w, loc=loc, scale_tril=scale_tril
    )

    # compute gradient
    if compute_gradient:
        log_density_grad, log_responsibilities, prec_times_diff = _gmm_log_density_grad(
            z=z,
            log_w=log_w,
            loc=loc,
            prec=prec,
            log_density=log_density,
            log_component_densities=log_component_densities,
        )
    else:
        log_density_grad = None

    # compute Hessian
    if compute_hessian:
        log_density_hess = _gmm_log_density_hess(
            prec=prec,
            log_responsibilities=log_responsibilities,
            prec_times_diff=prec_times_diff,
            log_density_grad=log_density_grad,
        )
    else:
        log_density_hess = None

    # check output
    assert log_density.shape == (n_samples,) + batch_shape
    if compute_gradient:
        assert log_density_grad.shape == (n_samples,) + batch_shape + (d_z,)
    if compute_hessian:
        assert log_density_hess.shape == (n_samples,) + batch_shape + (d_z, d_z)
    return log_density, log_density_grad, log_density_hess


@tf.function
def _gmm_log_density_and_log_component_densities(
    z: tf.Tensor, log_w: tf.Tensor, loc: tf.Tensor, scale_tril: tf.Tensor
):
    # check input
    n_samples = z.shape[0]
    d_z = loc.shape[-1]
    n_components = loc.shape[-2]
    batch_shape = loc.shape[:-2]
    assert z.shape == (n_samples,) + batch_shape + (d_z,)
    assert log_w.shape == batch_shape + (n_components,)
    assert loc.shape == batch_shape + (n_components, d_z)
    assert scale_tril.shape == batch_shape + (n_components, d_z, d_z)

    # compute log component densities and log density
    log_component_densities = gmm_log_component_densities(
        z=z,
        loc=loc,
        scale_tril=scale_tril,
    )
    log_density = gmm_log_density(
        z=z,
        log_w=log_w,
        log_component_densities=log_component_densities,
    )

    # check output
    assert log_density.shape == (n_samples,) + batch_shape
    assert log_component_densities.shape == (n_samples,) + batch_shape + (n_components,)
    return log_density, log_component_densities


@tf.function
def _gmm_log_density_grad(
    z: tf.Tensor,
    log_w: tf.Tensor,
    loc: tf.Tensor,
    prec: tf.Tensor,
    log_density: tf.Tensor,
    log_component_densities: tf.Tensor,
):
    # check input
    n_samples = z.shape[0]
    d_z = loc.shape[-1]
    n_components = loc.shape[-2]
    batch_shape = loc.shape[:-2]
    assert z.shape == (n_samples,) + batch_shape + (d_z,)
    assert log_w.shape == batch_shape + (n_components,)
    assert loc.shape == batch_shape + (n_components, d_z)
    assert prec.shape == batch_shape + (n_components, d_z, d_z)
    assert log_density.shape == (n_samples,) + batch_shape
    assert log_component_densities.shape == (n_samples,) + batch_shape + (n_components,)

    # compute d/dz log q(z)
    log_responsibilities = gmm_log_responsibilities(
        z=z,
        log_w=log_w,
        log_density=log_density,
        log_component_densities=log_component_densities,
    )
    assert log_responsibilities.shape == (n_samples,) + batch_shape + (n_components,)
    prec_times_diff = loc[None, ...] - z[..., None, :]
    assert prec_times_diff.shape == (n_samples,) + batch_shape + (n_components, d_z)
    prec_times_diff = tf.einsum("...ij,s...j->s...i", prec, prec_times_diff)
    assert prec_times_diff.shape == (n_samples,) + batch_shape + (n_components, d_z)
    # sum over components
    log_density_grad, signs = tfp.math.reduce_weighted_logsumexp(
        logx=log_responsibilities[..., None]
        + tf.math.log(tf.math.abs(prec_times_diff)),
        w=tf.math.sign(prec_times_diff),
        axis=-2,
        return_sign=True,
    )
    log_density_grad = tf.math.exp(log_density_grad) * signs

    # check output
    assert log_density_grad.shape == (n_samples,) + batch_shape + (d_z,)
    assert log_responsibilities.shape == (n_samples,) + batch_shape + (n_components,)
    assert prec_times_diff.shape == (n_samples,) + batch_shape + (n_components, d_z)
    return log_density_grad, log_responsibilities, prec_times_diff


@tf.function
def _gmm_log_density_hess(
    prec: tf.Tensor,
    log_responsibilities: tf.Tensor,
    prec_times_diff: tf.Tensor,
    log_density_grad: tf.Tensor,
):
    # check input
    n_samples = log_responsibilities.shape[0]
    n_components = log_responsibilities.shape[-1]
    batch_shape = log_responsibilities.shape[1:-1]
    d_z = log_density_grad.shape[-1]
    assert prec.shape == batch_shape + (n_components, d_z, d_z)
    assert log_responsibilities.shape == (n_samples,) + batch_shape + (n_components,)
    assert prec_times_diff.shape == (n_samples,) + batch_shape + (n_components, d_z)
    assert log_density_grad.shape == (n_samples,) + batch_shape + (d_z,)

    # compute Hessian
    log_density_hess = prec_times_diff - log_density_grad[..., None, :]
    assert log_density_hess.shape == (n_samples,) + batch_shape + (n_components, d_z)
    log_density_hess = tf.einsum(
        "s...ki,s...kj->s...kij", log_density_hess, prec_times_diff
    )
    assert log_density_hess.shape == (n_samples,) + batch_shape + (
        n_components,
        d_z,
        d_z,
    )
    log_density_hess = log_density_hess - prec[None, ...]
    assert log_density_hess.shape == (n_samples,) + batch_shape + (
        n_components,
        d_z,
        d_z,
    )
    # sum over components
    log_density_hess, signs = tfp.math.reduce_weighted_logsumexp(
        logx=log_responsibilities[..., None, None]
        + tf.math.log(tf.math.abs(log_density_hess)),
        w=tf.math.sign(log_density_hess),
        axis=-3,
        return_sign=True,
    )
    log_density_hess = tf.math.exp(log_density_hess) * signs

    # check output
    assert log_density_hess.shape == (n_samples,) + batch_shape + (d_z, d_z)
    return log_density_hess


def eval_fn_grad_hess(fn, z: tf.Tensor, compute_grad: bool, compute_hess: bool):
    # check input
    assert not (compute_grad == False and compute_hess == True)
    n_samples = z.shape[0]
    batch_shape = z.shape[1:-1]
    d_z = z.shape[-1]
    assert z.shape == (n_samples,) + batch_shape + (d_z,)

    if not compute_grad:
        f_z, f_z_grad, f_z_hess = _eval_fn(fn=fn, z=z), None, None
    elif compute_grad and not compute_hess:
        (f_z, f_z_grad), f_z_hess = _eval_fn_grad(fn=fn, z=z), None
    else:
        f_z, f_z_grad, f_z_hess = _eval_fn_grad_hess(fn=fn, z=z)

    # check output
    assert f_z.shape == (n_samples,) + batch_shape
    if compute_grad:
        assert f_z_grad.shape == (n_samples,) + batch_shape + (d_z,)
    if compute_hess:
        assert f_z_hess.shape == (n_samples,) + batch_shape + (d_z, d_z)

    return f_z, f_z_grad, f_z_hess


@tf.function
def _eval_fn(fn, z: tf.Tensor):
    # compute fn(z)
    f_z = fn(z)
    return f_z


@tf.function
def _eval_fn_grad(fn, z: tf.Tensor):
    # compute fn(z), and its gradient
    with tf.GradientTape(watch_accessed_variables=False) as g2:
        g2.watch(z)
        f_z = fn(z)
    f_z_grad = g2.gradient(f_z, z)

    return f_z, f_z_grad


@tf.function
def _eval_fn_grad_hess(fn, z: tf.Tensor):
    # TODO: fully batch the Hessian computation
    #  The problem is that batch_jacobian allows only one batch dimension. If
    #  multiple batch dimensions (e.g., sample-dim and task-dim) are present,
    #  batch jacobian does unneccesary computations akin to the ones described
    #  here: https://www.tensorflow.org/guide/advanced_autodiff#batch_jacobian
    #  Flattening the batch-dims **after** gradient computation and before calling
    #  batch jacobian is not an option as tf creates new variables for reshaped
    #  variables, losing track of gradients.
    #  Flattening the batch-dims **before** all computations is not an option
    #  because fn(z) has to be called with correct batch dims, as in general
    #  it performs different computations for each batch dim.

    # compute fn(z), and its gradient and hessian

    # check input
    n_samples = z.shape[0]
    batch_shape = z.shape[1:-1]
    d_z = z.shape[-1]
    assert not len(batch_shape) > 1  # not implemented yet

    # # flatten batch dimensions
    # z = tf.reshape(z, (-1, d_z))

    with tf.GradientTape(watch_accessed_variables=False, persistent=True) as g1:
        g1.watch(z)
        with tf.GradientTape(watch_accessed_variables=False) as g2:
            g2.watch(z)
            f_z = fn(z)
        f_z_grad = g2.gradient(f_z, z)
    f_z_hess = g1.batch_jacobian(f_z_grad, z)  # TODO: performs unneccesary computations
    if len(batch_shape) == 1:
        f_z_hess = tf.einsum("sbxby->sbxy", f_z_hess)

    # unflatten batch dimensions
    # f_z = tf.reshape(f_z, (n_samples,) + batch_shape)
    # f_z_grad = tf.reshape(f_z_grad, (n_samples,) + batch_shape + (d_z,))
    # f_z_hess = tf.reshape(f_z_hess, (n_samples,) + batch_shape + (d_z, d_z))

    return f_z, f_z_grad, f_z_hess


@tf.function
def expectation_prod_neg(log_a_z: tf.Tensor, b_z: tf.Tensor):
    """
    Compute the expectation E_q[a(z) * b(z)] \approx 1/S \sum_s a(z_s) b(z_s) in a
    stable manner, where z_s is sampled from q, and S is the number of samples.
    b(z_s) may contain negative or zero entries. The 0th dimension is assumed to
    be the sample dimension.
    """
    ## check inputs
    n_samples = log_a_z.shape[0]
    assert b_z.shape[0] == n_samples

    ## compute expectation
    expectation, signs = tfp.math.reduce_weighted_logsumexp(
        log_a_z + tf.math.log(tf.math.abs(b_z)),
        w=tf.math.sign(b_z),
        axis=0,
        return_sign=True,
    )
    expectation = tf.exp(expectation) * signs
    expectation = 1.0 / n_samples * expectation

    # check output
    # not allowed in graph mode
    # assert expectation.shape == tf.broadcast_static_shape(log_a_z.shape, b_z.shape)[1:]
    return expectation


@tf.function
def compute_S_bar(
    z: tf.Tensor, mu: tf.Tensor, prec: tf.Tensor, log_tgt_post_grad: tf.Tensor
):
    """
    compute S_bar = prec * (z - mu) * (-log_tgt_post_grad)
    """
    # check input
    n_samples = z.shape[0]
    d_z = mu.shape[-1]
    batch_shape = z.shape[1:-1]
    n_components = mu.shape[-2]
    assert z.shape == (n_samples,) + batch_shape + (d_z,)
    assert mu.shape == batch_shape + (n_components, d_z)
    assert prec.shape == batch_shape + (n_components, d_z, d_z)
    assert log_tgt_post_grad.shape == (n_samples,) + batch_shape + (d_z,)

    S_bar = z[..., None, :] - mu[None, ..., :, :]
    assert S_bar.shape == (n_samples,) + batch_shape + (n_components, d_z)
    S_bar = tf.einsum("...kij,s...kj->s...ki", prec, S_bar)
    assert S_bar.shape == (n_samples,) + batch_shape + (n_components, d_z)
    S_bar = tf.einsum("s...ki,s...j->s...kij", S_bar, -log_tgt_post_grad)

    # check output
    assert S_bar.shape == (n_samples,) + batch_shape + (n_components, d_z, d_z)
    return S_bar


@tf.function
def log_w_to_log_omega(log_w: tf.Tensor):
    """
    Compute log_omega from log_w.
    -> log_omega[k] := log(w[k]/w[K]), k = 1...K-1
    """
    # check input
    n_components = log_w.shape[-1]

    # compute log_omega
    log_omega = log_w[..., :-1] - log_w[..., -1:]

    # check output
    assert log_omega.shape[-1:] == (n_components - 1,)
    return log_omega


@tf.function
def log_omega_to_log_w(log_omega: tf.Tensor):
    """
    Compute log_w from log_omega.
    -> log_w[k] = log_omega[k]  - logsumexp(log_omega_tilde) for k=1..K-1
       with log_omega_tilde[k] = log_omega[k] (for k=1..K-1)
            log_omega_tilde[k] = 0            (for k=K)
    """
    # check input
    n_components = log_omega.shape[-1] + 1

    # compute log_omega_tilde
    log_omega_tilde = tf.concat(
        (log_omega, tf.zeros(log_omega.shape[:-1] + (1,))), axis=-1
    )

    # compute log_w
    lse_log_omega_tilde = tf.reduce_logsumexp(log_omega_tilde, keepdims=True, axis=-1)
    log_w = log_omega - lse_log_omega_tilde
    log_w = tf.concat((log_w, -lse_log_omega_tilde), axis=-1)

    # check output
    assert log_w.shape[-1:] == (n_components,)
    return log_w


class GMM:
    def __init__(self, log_w: tf.Tensor, loc: tf.Tensor, prec: tf.Tensor):
        # check input
        self.n_components = log_w.shape[-1]
        self.d_z = loc.shape[-1]
        assert log_w.shape[-1:] == (self.n_components,)
        assert loc.shape[-2:] == (self.n_components, self.d_z)
        assert prec.shape[-3:] == (self.n_components, self.d_z, self.d_z)

        self._log_w = tf.Variable(tf.cast(log_w, dtype=tf.float32))
        self._loc = tf.Variable(tf.cast(loc, dtype=tf.float32))
        self._prec = tf.Variable(tf.cast(prec, dtype=tf.float32))
        self._prec_tril = tf.Variable(prec_to_prec_tril(prec=self._prec))
        self._scale_tril = tf.Variable(prec_to_scale_tril(prec=self._prec))
        self._cov = tf.Variable(scale_tril_to_cov(self._scale_tril))

    @property
    def loc(self):
        return self._loc

    @property
    def log_w(self):
        return self._log_w

    @property
    def prec(self):
        return self._prec

    @property
    def prec_tril(self):
        return self._prec_tril

    @property
    def scale_tril(self):
        return self._scale_tril

    @property
    def cov(self):
        return self._cov

    # TODO: check consistency of shapes
    @loc.setter
    def loc(self, value):
        self._loc.assign(value)

    @log_w.setter
    def log_w(self, value):
        self._log_w.assign(value)

    @prec.setter
    def prec(self, value):
        self._prec.assign(value)
        self._prec_tril.assign(prec_to_prec_tril(self._prec))
        self._scale_tril.assign(prec_to_scale_tril(self.prec))
        self._cov.assign(scale_tril_to_cov(self.scale_tril))

    @prec_tril.setter
    def prec_tril(self, value):
        raise NotImplementedError

    @scale_tril.setter
    def scale_tril(self, value):
        raise NotImplementedError

    @cov.setter
    def cov(self, value):
        raise NotImplementedError

    @tf.function
    def sample(self, n_samples: int):
        return sample_gmm(
            n_samples=n_samples,
            log_w=self.log_w,
            loc=self.loc,
            scale_tril=self.scale_tril,
        )

    @tf.function
    def log_density(self, z: tf.Tensor):
        return gmm_log_density(
            z=z, log_w=self.log_w, loc=self.loc, scale_tril=self.scale_tril
        )

    @tf.function
    def log_component_densities(self, z: tf.Tensor):
        return gmm_log_component_densities(
            z=z, loc=self.loc, scale_tril=self.scale_tril
        )
