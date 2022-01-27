import torch

# TODO: use the following functions?
# def sample_categorical(num_samples, log_w):
#     thresholds = tf.expand_dims(tf.cumsum(tf.exp(log_w)), 0)
#     #  thresholds[0, -1] = 1.0
#     eps = tf.random.uniform(shape=[num_samples, 1])
#     samples = tf.argmax(eps < thresholds, axis=-1, output_type=tf.int32)
#     return samples


# def sample_Gaussian(dim, mean, chol, num_samples):
#     return tf.transpose(
#         tf.expand_dims(mean, axis=-1)
#         + chol @ tf.random.normal((dim, num_samples), mean=0.0, stddev=1.0)
#     )


# def gaussian_log_pdf(dim, mean, chol, inv_chol, x):
#     constant_part = -0.5 * dim * tf.math.log(2 * pi) - tf.reduce_sum(
#         tf.math.log(tf.linalg.diag_part(chol))
#     )
#     return constant_part - 0.5 * tf.reduce_sum(
#         tf.square(inv_chol @ tf.transpose(mean - x)), axis=0
#     )


# def eval_fn_grad_hess(
#     fn,
#     z: torch.Tensor,
#     compute_grad: bool,
#     compute_hess: bool,
# ):
#     # check input
#     n_tasks = z.shape[0]
#     n_samples = z.shape[1]
#     d_z = z.shape[2]
#     assert z.shape == (n_samples, n_tasks, d_z)
#     assert not (compute_grad == False and compute_hess == True)

#     # eval function, and compute gradient and hessian if necessary
#     if not compute_grad:
#         f_z, f_z_grad, f_z_hess = _eval_fun(fun=fn, z=z), None, None
#     elif compute_grad and not compute_hess:
#         (f_z, f_z_grad), f_z_hess = _eval_fun_grad(fun=fn, z=z), None
#     else:
#         assert compute_grad and compute_hess
#         f_z, f_z_grad, f_z_hess = _eval_fun_grad_hess(fun=fn, z=z)

#     # check output
#     assert f_z.shape == (n_samples, n_tasks)
#     if compute_grad:
#         assert f_z_grad.shape == (n_samples, n_tasks, d_z)
#     if compute_hess:
#         assert f_z_hess.shape == (n_samples, n_tasks, d_z, d_z)
#     return f_z, f_z_grad, f_z_hess


def eval_fn_grad_hess(
    fn,
    z: torch.Tensor,
    compute_grad: bool,
    compute_hess: bool,
):
    # check input
    n_samples = z.shape[0]
    n_tasks = z.shape[1]
    d_z = z.shape[2]
    assert z.shape == (n_samples, n_tasks, d_z)
    assert not (compute_grad == False and compute_hess == True)

    # eval function, and compute gradient and hessian if necessary
    # z.requires_grad_(True)
    f_z = fn(z)
    f_z_grad = (
        torch.autograd.functional.jacobian(
            func=fn, inputs=z, vectorize=True, create_graph=compute_hess
        )
        if compute_grad
        else None
    )
    f_z_hess = (
        torch.autograd.functional.hessian(func=fn, inputs=z) if compute_hess else None
    )

    # check output
    assert f_z.shape == (n_samples, n_tasks)
    if compute_grad:
        assert f_z_grad.shape == (n_samples, n_tasks, d_z)
    if compute_hess:
        assert f_z_hess.shape == (n_samples, n_tasks, d_z, d_z)
    return f_z, f_z_grad, f_z_hess


# def _eval_fun(fun, z: torch.Tensor):
#     # compute fun(z)
#     f_z = fun(z)
#     return f_z


# def _eval_fun_grad(fun, z: torch.Tensor):
#     # compute fun(z), and its gradient
#     with tf.GradientTape(watch_accessed_variables=False) as g2:
#         g2.watch(z)
#         f_z = fun(z)
#     f_z_grad = g2.gradient(f_z, z)

#     return f_z, f_z_grad


# def _eval_fun_grad_hess(fun, z: torch.Tensor):
#     # compute fun(z), and its gradient and hessian
#     with tf.GradientTape(watch_accessed_variables=False) as g1:
#         g1.watch(z)
#         with tf.GradientTape(watch_accessed_variables=False) as g2:
#             g2.watch(z)
#             f_z = fun(z)
#         f_z_grad = g2.gradient(f_z, z)
#     f_z_hess = g1.batch_jacobian(f_z_grad, z)

#     return f_z, f_z_grad, f_z_hess


def expectation_prod_neg(log_a_z: torch.Tensor, b_z: torch.Tensor):
    """
    Compute the expectation E_q[a(z) * b(z)] \approx 1/S \sum_s a(z_s) b(z_s) in a
    stable manner, where z_s is sampled from q, and S is the number of samples.
    b(z_s) may contain negative or zero entries.
    """
    ## check inputs
    n_samples = b_z.shape[0]
    assert b_z.shape[0] == log_a_z.shape[0]

    ## compute expectation
    expectation, signs = weighted_logsumexp(
        log_a_z + torch.log(torch.abs(b_z)),
        w=torch.sign(b_z),
        dim=0,
        return_sign=True,
    )
    expectation = torch.exp(expectation) * signs
    expectation = 1 / n_samples * expectation

    # check output
    assert expectation.shape == torch.broadcast_shapes(log_a_z.shape, b_z.shape)[1:]
    return expectation


def weighted_logsumexp(
    logx: torch.tensor,
    w: torch.tensor,
    dim: int,
    keepdim: bool = False,
    return_sign: bool = False,
):
    """
    Adapted from tensorflow's tf.math.reduce_weighted_logsumexp.
    """

    log_absw_x = logx + torch.log(torch.abs(w))
    max_log_absw_x, _ = torch.max(log_absw_x, dim=dim, keepdim=True)
    # If the largest element is `-inf` or `inf` then we don't bother subtracting
    # off the max. We do this because otherwise we'd get `inf - inf = NaN`. That
    # this is ok follows from the fact that we're actually free to subtract any
    # value we like, so long as we add it back after taking the `log(sum(...))`.
    max_log_absw_x = torch.where(
        torch.isinf(max_log_absw_x),
        torch.tensor(0.0, dtype=max_log_absw_x.dtype),
        max_log_absw_x,
    )
    wx_over_max_absw_x = torch.sign(w) * torch.exp(log_absw_x - max_log_absw_x)
    sum_wx_over_max_absw_x = torch.sum(
        wx_over_max_absw_x,
        dim=dim,
        keepdim=keepdim,
    )
    if not keepdim:
        max_log_absw_x = torch.squeeze(max_log_absw_x, dim)
    sgn = torch.sign(sum_wx_over_max_absw_x)
    lswe = max_log_absw_x + torch.log(sgn * sum_wx_over_max_absw_x)
    if return_sign:
        return lswe, sgn
    return lswe


def log_weights_to_log_omegas(log_weights: torch.Tensor):
    """
    Compute log_omega from log_w.
    -> log_omega[k] := log(w[k]/w[K]), k = 1...K-1
    """
    # check input
    n_tasks = log_weights.shape[0]
    n_components = log_weights.shape[1]
    assert log_weights.shape == (n_tasks, n_components)

    # compute log_omega
    log_omega = log_weights[:, :-1] - log_weights[:, -1:]

    # check output
    assert log_omega.shape == (n_tasks, n_components - 1)
    return log_omega


def log_omegas_to_log_weights(log_omegas: torch.Tensor):
    """
    Compute log_w from log_omega.
    -> log_w[k] = log_omega[k]  - logsumexp(log_omega_tilde) for k=1..K-1
       with log_omega_tilde[k] = log_omega[k] (for k=1..K-1)
            log_omega_tilde[k] = 0            (for k=K)
    """
    # check input
    n_tasks = log_omegas.shape[0]
    n_components = log_omegas.shape[1] + 1
    assert log_omegas.shape == (n_tasks, n_components - 1)

    # compute log_omega_tilde
    log_omega_tilde = torch.cat((log_omegas, torch.zeros(n_tasks, 1)), dim=1)

    # compute log_w
    lse_log_omega_tilde = torch.logsumexp(log_omega_tilde, dim=1, keepdim=True)
    log_weights = log_omegas - lse_log_omega_tilde
    log_weights = torch.cat((log_weights, -lse_log_omega_tilde), axis=1)

    # check output
    assert log_weights.shape == (n_tasks, n_components)
    return log_weights


def precs_to_prec_trils(precs: torch.Tensor):
    # Compute lower cholesky factors of precision matrices

    # check input
    n_tasks = precs.shape[0]
    n_components = precs.shape[1]
    d_z = precs.shape[2]
    assert precs.shape == (n_tasks, n_components, d_z, d_z)

    # from: https://pytorch.org/docs/stable/_modules/torch/distributions/multivariate_normal.html#MultivariateNormal
    prec_trils = torch.linalg.cholesky(precs)

    # check output
    assert prec_trils.shape == (n_tasks, n_components, d_z, d_z)
    return prec_trils


# def prec_to_scale_tril(prec: tf.Tensor):
#     # Compute lower cholesky factors of covariance matrices from precision matrices

#     # check input
#     # assert tf.rank(prec) == 3
#     K = prec.shape[0]
#     D = prec.shape[1]
#     # assert prec.shape[2] == D

#     # from: https://pytorch.org/docs/stable/_modules/torch/distributions/multivariate_normal.html#MultivariateNormal
#     Lf = tf.linalg.cholesky(tf.reverse(prec, axis=(-2, -1)))
#     L_inv = tf.transpose(tf.reverse(Lf, axis=(-2, -1)), perm=(0, 2, 1))
#     L = tf.linalg.triangular_solve(L_inv, tf.eye(D, dtype=L_inv.dtype), lower=True)

#     # check output
#     # assert L.shape == (K, D, D)
#     return L


# def scale_tril_to_cov(scale_tril: tf.Tensor):
#     # check input
#     # assert tf.rank(scale_tril) == 3
#     D = scale_tril.shape[1]
#     # assert scale_tril.shape[2] == D

#     # compute cov from scale_tril
#     cov = tf.matmul(scale_tril, tf.transpose(scale_tril, perm=(0, 2, 1)))

#     # check output
#     # assert cov.shape == scale_tril.shape
#     return cov


# def cov_to_scale_tril(cov: tf.Tensor):
#     # check input
#     # assert tf.rank(cov) == 3
#     D = cov.shape[1]
#     # assert cov.shape[2] == D

#     # compute scale_tril from cov
#     scale_tril = tf.linalg.cholesky(cov)

#     # check output
#     # assert scale_tril.shape == cov.shape
#     return scale_tril


def precs_to_covs(precs: torch.Tensor):
    # check input
    n_tasks = precs.shape[0]
    n_components = precs.shape[1]
    d_z = precs.shape[2]
    assert precs.shape == (n_tasks, n_components, d_z, d_z)

    # compute cov from prec
    # TODO: is there a better way? use prec_to_scale_tril?
    covs = torch.linalg.inv(precs)

    # check output
    assert covs.shape == (n_tasks, n_components, d_z, d_z)
    return precs


def covs_to_precs(covs: torch.Tensor):
    # check input
    n_tasks = covs.shape[0]
    n_components = covs.shape[1]
    d_z = covs.shape[2]
    assert covs.shape == (n_tasks, n_components, d_z, d_z)

    # compute cov from prec
    # TODO: is there a better way? use prec_to_scale_tril?
    prec = torch.linalg.inv(covs)

    # check output
    assert prec.shape == (n_tasks, n_components, d_z, d_z)
    return prec


# def compute_S_bar(z, mu, prec, log_tgt_post_grad):
#     """
#     compute S_bar = prec * (z - mu) * (-log_tgt_post_grad)
#     """
#     # check input
#     S = z.shape[0]
#     K = mu.shape[0]
#     D = mu.shape[1]
#     # assert z.shape == (S, D)
#     # assert mu.shape == (K, D)
#     # assert prec.shape == (K, D, D)
#     # assert log_tgt_post_grad.shape == (S, D)

#     S_bar = z[:, None, :] - mu[None, :, :]
#     # assert S_bar.shape == (S, K, D)
#     S_bar = tf.einsum("kij,skj->ski", prec, S_bar)
#     # assert S_bar.shape == (S, K, D)
#     S_bar = tf.einsum("ski,sj->skij", S_bar, -log_tgt_post_grad)

#     # check output
#     # assert S_bar.shape == (S, K, D, D)
#     return S_bar

# TODO: use the following class as the latent distribution
# class GMM:
#     def __init__(self, log_w: tf.Tensor, loc: tf.Tensor, prec: tf.Tensor):
#         # check input
#         # assert tf.rank(log_w) == 1
#         # assert tf.rank(loc) == 2
#         # assert tf.rank(prec) == 3
#         self.K = log_w.shape[0]
#         self.D = loc.shape[1]
#         # assert loc.shape[0] == prec.shape[0] == self.K
#         # assert prec.shape[1] == prec.shape[2] == self.D

#         self.validate_args = False  # only set to True for debugging!
#         self._log_w = tf.Variable(tf.cast(log_w, dtype=tf.float32))
#         self._loc = tf.Variable(tf.cast(loc, dtype=tf.float32))
#         self._prec = tf.Variable(tf.cast(prec, dtype=tf.float32))
#         self._prec_tril = tf.Variable(prec_to_prec_tril(prec=self._prec))
#         self._scale_tril = tf.Variable(prec_to_scale_tril(prec=self._prec))
#         self._cov = tf.Variable(scale_tril_to_cov(self._scale_tril))

#         # self.gmm = tfp.distributions.MixtureSameFamily(
#         #     mixture_distribution=tfp.distributions.Categorical(
#         #         logits=self._log_w,
#         #         validate_args=self.validate_args,
#         #     ),
#         #     components_distribution=tfp.distributions.MultivariateNormalTriL(
#         #         loc=self._loc,
#         #         # TODO: it would be better to use prec_tril!
#         #         scale_tril=self._scale_tril,
#         #         validate_args=self.validate_args,
#         #     ),
#         #     validate_args=self.validate_args,
#         # )

#     @property
#     def loc(self):
#         return self._loc

#     @property
#     def log_w(self):
#         return self._log_w

#     @property
#     def prec(self):
#         return self._prec

#     @property
#     def prec_tril(self):
#         return self._prec_tril

#     @property
#     def scale_tril(self):
#         return self._scale_tril

#     @property
#     def cov(self):
#         return self._cov

#     @loc.setter
#     def loc(self, value):
#         self._loc.assign(value)

#     @log_w.setter
#     def log_w(self, value):
#         self._log_w.assign(value)

#     @prec.setter
#     def prec(self, value):
#         self._prec.assign(value)
#         self._prec_tril.assign(prec_to_prec_tril(self._prec))
#         self._scale_tril.assign(prec_to_scale_tril(self.prec))
#         self._cov.assign(scale_tril_to_cov(self.scale_tril))

#     @prec_tril.setter
#     def prec_tril(self, value):
#         raise NotImplementedError

#     @scale_tril.setter
#     def scale_tril(self, value):
#         raise NotImplementedError

#     @cov.setter
#     def cov(self, value):
#         raise NotImplementedError

#     @tf.function  # TODO: why does this give a speedup?
#     def sample(self, n_samples: int):
#         sampled_components = sample_categorical(num_samples=n_samples, log_w=self.log_w)
#         samples = []
#         for i in range(self.K):
#             n_samples = tf.reduce_sum(tf.cast(sampled_components == i, tf.int32))
#             this_samples = sample_Gaussian(
#                 self.D, self.loc[i], self.scale_tril[i], n_samples
#             )
#             samples.append(this_samples)
#         return tf.random.shuffle(tf.concat(samples, axis=0))

#     @tf.function
#     def log_density(self, z: tf.Tensor):
#         log_densities = self.log_component_densities(z)
#         weighted_densities = log_densities + tf.expand_dims(self.log_w, axis=0)
#         return tf.reduce_logsumexp(weighted_densities, axis=1)

#     def log_density_grad_hess(
#         self, z: tf.Tensor, compute_grad: bool, compute_hess: bool, test_fun=None
#     ):
#         """
#         Compute the log density, (i.e. q(z)), and its gradient and hessian.
#         """
#         fun = self.log_density if test_fun is None else test_fun
#         log_density, log_density_grad, log_density_hess = eval_grad_hess(
#             fn=fun, z=z, compute_grad=compute_grad, compute_hess=compute_hess
#         )
#         return log_density, log_density_grad, log_density_hess

#     @tf.function  # TODO: why does this give a speedup (not used in autograd!)
#     def log_component_densities(self, z: tf.Tensor):
#         diffs = tf.expand_dims(z, 0) - tf.expand_dims(self.loc, 1)
#         sqrts = tf.linalg.triangular_solve(
#             self.scale_tril, tf.transpose(diffs, [0, 2, 1])
#         )
#         mahalas = -0.5 * tf.reduce_sum(sqrts * sqrts, axis=1)
#         const_parts = -0.5 * tf.reduce_sum(
#             tf.math.log(tf.square(tf.linalg.diag_part(self.scale_tril))), axis=1
#         ) - 0.5 * self.D * tf.math.log(2 * pi)
#         return tf.transpose(mahalas + tf.expand_dims(const_parts, axis=1))
