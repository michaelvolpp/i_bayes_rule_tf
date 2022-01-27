import math

import pytest
import torch
from aggbyopt_nps.util import (
    eval_fn_grad_hess,
    expectation_prod_neg,
    log_omegas_to_log_weights,
    log_weights_to_log_omegas,
)


def test_expectation_prod_neg():
    a_z = torch.tensor(
        [
            [
                [1.0, 2.0],
                [1.0, 5.0],
            ],
            [
                [3.0, 4.0],
                [3.5, 0.1],
            ],
            [
                [5.0, 6.0],
                [0.0001, 60000.0],
            ],
            [
                [5.0, 6.0],
                [5000.0, 6.0],
            ],
            [
                [5.0, 6.0],
                [5.0, 6.0],
            ],
        ]
    )
    b_z = torch.tensor(
        [
            [
                [3.0, 1.0, -3.0],
                [30.0, 10.0, -300.0],
            ],
            [
                [-2.0, 1.0, 0.0],
                [-2.00001, 1.0001, 0.0001],
            ],
            [
                [1.0, 2.0, -17.0],
                [100.0, -200.0, -17.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [-190.0, -200.0, -1002],
                [-1900.0, 200.0, -1002],
            ],
        ]
    )
    assert (a_z > 0.0).all()
    n_samples = a_z.shape[0]
    n_tasks = a_z.shape[1]
    assert b_z.shape[0] == n_samples
    assert b_z.shape[1] == n_tasks

    # compute ground truth
    true_expectation = torch.mean(a_z[:, :, :, None] * b_z[:, :, None, :], axis=0)
    assert true_expectation.shape == (
        n_tasks,
        a_z.shape[2],
        b_z.shape[2],
    )

    # check that computation is correct
    computed_expectation = expectation_prod_neg(
        log_a_z=torch.log(a_z[:, :, :, None]), b_z=b_z[:, :, None, :]
    )
    assert computed_expectation.shape == (
        n_tasks,
        a_z.shape[2],
        b_z.shape[2],
    )
    assert torch.allclose(true_expectation, computed_expectation)


# def test_compute_S_bar():
#     # generate relevant tensors
#     S = 5
#     K = 4
#     D = 3
#     z = tf.random.uniform(shape=(S, D))
#     mu = tf.random.uniform(shape=(K, D))
#     # for this test, we do not require prec to be a valid precision matrix
#     prec = tf.random.uniform(shape=(K, D, D))
#     log_tgt_post_grad = tf.random.uniform(shape=(S, D))

#     # check computation
#     S_bar = compute_S_bar(z=z, mu=mu, prec=prec, log_tgt_post_grad=log_tgt_post_grad)
#     for s in range(S):
#         for k in range(K):
#             for i in range(D):
#                 for j in range(D):
#                     S_bar_skij = prec[k] @ (z[s] - mu[k])[:, None]
#                     S_bar_skij = S_bar_skij[i] * log_tgt_post_grad[s, j]
#                     assert S_bar[s, k, i, j] == S_bar_skij


def test_log_omegas_log_weights_conversion():
    ## check that log_w == log_omega_to_log_w(log_w_to_log_omega(log_w))
    # generate w
    weights = torch.tensor([[0.1, 0.2, 0.3, 0.2], [0.8, 0.01, 0.001, 0.1]])
    weights = torch.cat((weights, 1.0 - torch.sum(weights, dim=1, keepdim=True)), dim=1)
    assert (torch.sum(weights, dim=1) == 1.0).all()
    # perform test
    log_weights = torch.log(weights)
    log_omegas = log_weights_to_log_omegas(log_weights)
    assert torch.allclose(
        log_omegas, torch.log(weights[:, :4]) - torch.log(weights[:, 4:5])
    )
    assert torch.allclose(log_weights, log_omegas_to_log_weights(log_omegas))

    ## check that log_omega == log_w_to_log_omega(log_omega_to_log_w(log_omega))
    # generate omega
    log_omegas = torch.log(weights[:, :4]) - torch.log(weights[:, 4:5])
    # perform test
    log_weights = log_omegas_to_log_weights(log_omegas)
    assert torch.allclose(log_omegas, log_weights_to_log_omegas(log_weights))


# def test_prec_to_prec_tril():
#     # low D
#     L_true = tf.constant(
#         [
#             [[1.0, 0.0], [2.0, 1.0]],
#             [[3.0, 0.0], [-7.0, 3.5]],
#             [[math.pi, 0.0], [math.e, 124]],
#         ]
#     )
#     prec = L_true @ tf.transpose(L_true, perm=(0, 2, 1))
#     L_comp = prec_to_prec_tril(prec=prec)
#     assert tf.experimental.numpy.allclose(L_comp, L_true)

#     # high D: requires 64-bit precision to pass test
#     tf.random.set_seed(123)
#     L_true = tf.stack(
#         [
#             tf.experimental.numpy.tril(tf.random.uniform([10, 10], dtype=tf.float64)),
#             tf.experimental.numpy.tril(tf.random.uniform([10, 10], dtype=tf.float64)),
#         ]
#     )
#     prec = L_true @ tf.transpose(L_true, perm=(0, 2, 1))
#     L_comp = prec_to_prec_tril(prec=prec)
#     assert tf.experimental.numpy.allclose(L_comp, L_true)


# def test_prec_to_scale_tril():
#     # low D
#     L = tf.constant(
#         [
#             [[1.0, 0.0], [2.0, 1.0]],
#             [[3.0, 0.0], [-7.0, 3.5]],
#             [[math.pi, 0.0], [math.e, 124]],
#         ]
#     )
#     cov = L @ tf.transpose(L, perm=(0, 2, 1))
#     prec = tf.linalg.inv(cov)
#     scale_tril = prec_to_scale_tril(prec=prec)
#     assert tf.experimental.numpy.allclose(scale_tril, L)

#     # high D: requires 64-bit precision to pass test
#     tf.random.set_seed(123)
#     L = tf.stack(
#         [
#             tf.experimental.numpy.tril(tf.random.uniform([10, 10], dtype=tf.float64)),
#             tf.experimental.numpy.tril(tf.random.uniform([10, 10], dtype=tf.float64)),
#         ]
#     )
#     cov = L @ tf.transpose(L, perm=(0, 2, 1))
#     prec = tf.linalg.inv(cov)
#     scale_tril = prec_to_scale_tril(prec=prec)
#     assert tf.experimental.numpy.allclose(scale_tril, L)


# def test_scale_tril_to_cov():
#     scale_tril = tf.constant(
#         [
#             [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
#             [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
#             [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
#         ]
#     )
#     true_cov = tf.constant(
#         [
#             [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
#             [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
#             [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
#         ]
#     )
#     cov = scale_tril_to_cov(scale_tril=scale_tril)
#     assert tf.experimental.numpy.allclose(cov, true_cov)


# def test_cov_to_scale_tril():
#     cov = tf.constant(
#         [
#             [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
#             [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
#             [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
#         ]
#     )
#     true_scale_tril = tf.constant(
#         [
#             [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
#             [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
#             [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
#         ]
#     )
#     scale_tril = cov_to_scale_tril(cov=cov)
#     assert tf.experimental.numpy.allclose(scale_tril, true_scale_tril)


def fn_with_known_grad_hess(z: torch.Tensor):
    # Implement f: R^2 -> R, f(z) = z_0**2 + 2*z_1**3 + 3*z_0*z_1 + 5
    n_samples = z.shape[0]
    n_tasks = z.shape[1]
    d_z = z.shape[2]
    assert d_z == 2
    assert z.shape == (n_samples, n_tasks, d_z)

    ## compute function value
    f_z = z[..., 0] ** 2 + 2 * z[..., 1] ** 3 + 3 * z[..., 0] * z[..., 1] + 5

    ## compute gradient
    f_z_grad = torch.stack(
        [
            # f_z_grad_0 = df/dz_0 = 2*z_0 + 3*z_1
            2 * z[..., 0] + 3 * z[..., 1],
            # f_z_grad_1 = df/dz_1 = 6*z_1**2 + 3*z_0
            6 * z[..., 1] ** 2 + 3 * z[..., 0],
        ],
        axis=2,
    )

    ## compute hessian
    f_z_hess = torch.stack(
        [
            # f_z_hess_00 = d^2f/dz_0^2 = 2
            # f_z_hess_01 = d^2f/dz_0 dz_1 = 3
            torch.stack(
                [
                    2 * torch.ones(n_samples, n_tasks),
                    3 * torch.ones(n_samples, n_tasks),
                ],
                axis=2,
            ),
            # f_z_hess_10 = d^2f/dz_1 dz_0 = 3
            # f_z_hess_11 = d^2f/dz_1^2 = 12 * z_1
            torch.stack(
                [
                    3 * torch.ones(n_samples, n_tasks),
                    12 * z[..., 1],
                ],
                axis=2,
            ),
        ],
        axis=3,
    )

    return f_z, f_z_grad, f_z_hess


def test_eval_fn_grad_hess():
    # generate samples
    n_samples = 10
    n_tasks = 5
    d_z = 2
    z = torch.rand((n_samples, n_tasks, d_z))

    ## test computation of gradients and hessian using a known test function
    log_density, log_density_grad, log_density_hess = eval_fn_grad_hess(
        fn=lambda z: fn_with_known_grad_hess(z)[0],
        z=z,
        compute_grad=True,
        compute_hess=False,
    )
    assert log_density.shape == (n_samples, n_tasks)
    assert log_density_grad.shape == (n_samples, n_tasks, d_z)
    assert log_density_hess.shape == (n_samples, n_tasks, d_z, d_z)
    (
        true_log_density,
        true_log_density_grad,
        true_log_density_hess,
    ) = fn_with_known_grad_hess(z)
    # TODO:
    # the values should match exactly, but for some runs they do and for others there
    # are miniscule differences -> why?
    assert torch.allclose(log_density, true_log_density)
    assert torch.allclose(log_density_grad, true_log_density_grad)
    # assert torch.allclose(log_density_hess, true_log_density_hess)


# def test_gmm():
#     # generate GMM
#     log_w = tf.math.log([0.1, 0.5, 0.4])
#     mu = tf.constant(
#         [
#             [-1.0, -1.0],
#             [-1.0, 1.0],
#             [1.0, 0.0],
#         ]
#     )
#     scale_tril = tf.constant(
#         [
#             [[0.1, 0.0], [0.0, 0.5]],
#             [[2.0, 0.0], [-0.5, 3.0]],
#             [[1.0, 0.0], [-4.0, 1.0]],
#         ]
#     )
#     prec = tf.linalg.inv(
#         tf.matmul(scale_tril, tf.transpose(scale_tril, perm=(0, 2, 1)))
#     )
#     K = log_w.shape[0]
#     D = mu.shape[1]
#     gmm = GMM(log_w=log_w, loc=mu, prec=prec)

#     ## test sample
#     tf.random.set_seed(124)
#     S = 10
#     z = gmm.sample(S)
#     assert z.shape == (S, D)
#     # set same seed again
#     tf.random.set_seed(124)
#     z2 = gmm.sample(S)
#     assert tf.reduce_all(z == z2)
#     # sample without reseeding
#     z3 = gmm.sample(S)
#     assert tf.reduce_all(z2 != z3)

#     ## test computation log_marginal_z using a test function with known grad/hess
#     log_density, log_density_grad, log_density_hess = gmm.log_density_grad_hess(
#         z=z,
#         compute_grad=True,
#         compute_hess=True,
#         test_fun=lambda z: fun_with_known_grad_hess(z)[0],
#     )
#     assert log_density.shape == (S,)
#     assert log_density_grad.shape == (S, D)
#     assert log_density_hess.shape == (S, D, D)
#     (
#         true_log_density,
#         true_log_density_grad,
#         true_log_density_hess,
#     ) = fun_with_known_grad_hess(z)
#     # TODO:
#     # the values should match exactly, but for some runs they do and for others there
#     # are miniscule differences -> why?
#     assert tf.experimental.numpy.allclose(log_density, true_log_density)
#     assert tf.experimental.numpy.allclose(log_density_grad, true_log_density_grad)
#     assert tf.experimental.numpy.allclose(log_density_hess, true_log_density_hess)

#     ## test computation log_marginal_z using the GMM likelihood
#     log_density, log_density_grad, log_density_hess = gmm.log_density_grad_hess(
#         z=z, compute_grad=True, compute_hess=True
#     )
#     assert log_density.shape == (S,)
#     assert log_density_grad.shape == (S, D)
#     assert log_density_hess.shape == (S, D, D)

#     log_density, log_density_grad, log_density_hess = gmm.log_density_grad_hess(
#         z=z, compute_grad=True, compute_hess=False
#     )
#     assert log_density.shape == (S,)
#     assert log_density_grad.shape == (S, D)
#     assert log_density_hess is None

#     log_density, log_density_grad, log_density_hess = gmm.log_density_grad_hess(
#         z=z, compute_grad=False, compute_hess=False
#     )
#     assert log_density.shape == (S,)
#     assert log_density_grad is None
#     assert log_density_hess is None

#     ## test computation of component distributions
#     # compute log_component_density with eager computation
#     tf.config.run_functions_eagerly(True)
#     log_comp_z = gmm.log_component_densities(z=z)
#     assert log_comp_z.shape == (S, K)
#     # compute log_component_density with graph-based computation
#     tf.config.run_functions_eagerly(False)
#     log_comp_z1 = gmm.log_component_densities(z=z)
#     assert tf.experimental.numpy.allclose(log_comp_z, log_comp_z1)
#     # compute log_component_density with graph-based computation and different params
#     gmm.loc = tf.random.normal(shape=mu.shape)
#     log_comp_z2 = gmm.log_component_densities(z=z)
#     assert tf.reduce_all(log_comp_z1 != log_comp_z2)

#     ## test repeated computation of log_density_grad_hess
#     S = 5
#     z = gmm.sample(S)
#     # choose a random mu
#     gmm.loc = tf.random.normal(shape=mu.shape)
#     # compute log_density with eager computation
#     tf.config.run_functions_eagerly(True)
#     log_density, log_density_grad, log_density_hess = gmm.log_density_grad_hess(
#         z=z, compute_grad=True, compute_hess=True
#     )
#     # compute log_density with graph-based computation
#     tf.config.run_functions_eagerly(False)
#     log_density1, log_density_grad1, log_density_hess1 = gmm.log_density_grad_hess(
#         z=z, compute_grad=True, compute_hess=True
#     )
#     # check that the results are the same
#     assert tf.experimental.numpy.allclose(log_density, log_density1)
#     assert tf.experimental.numpy.allclose(log_density_grad, log_density_grad1)
#     assert tf.experimental.numpy.allclose(log_density_hess, log_density_hess1)
#     # compute log_density with another random mu
#     gmm.loc = tf.random.normal(shape=mu.shape)
#     log_density2, log_density_grad2, log_density_hess2 = gmm.log_density_grad_hess(
#         z=z, compute_grad=True, compute_hess=True
#     )
#     # check that we obtain varying outputs:
#     assert tf.reduce_all(log_density1 != log_density2)
#     assert tf.reduce_all(log_density_grad1 != log_density_grad2)
#     # TODO: the hessians are the same sometimes, is this correct?
#     # assert tf.reduce_all(log_density_hess1 != log_density_hess2)


# def test_gmm_setters_getters():
#     ## generate GMM
#     log_w = tf.math.log([0.1, 0.5, 0.4])
#     mu = tf.constant(
#         [
#             [-1.0, -1.0],
#             [-1.0, 1.0],
#             [1.0, 0.0],
#         ]
#     )
#     scale_tril = tf.constant(
#         [
#             [[0.1, 0.0], [0.0, 0.5]],
#             [[2.0, 0.0], [-0.5, 3.0]],
#             [[1.0, 0.0], [-4.0, 1.0]],
#         ]
#     )
#     prec = tf.linalg.inv(
#         tf.matmul(scale_tril, tf.transpose(scale_tril, perm=(0, 2, 1)))
#     )
#     gmm = GMM(log_w=log_w, loc=mu, prec=prec)

#     # check that init works
#     assert tf.experimental.numpy.allclose(gmm.gmm.components_distribution.loc, mu)
#     assert tf.experimental.numpy.allclose(gmm.loc, mu)
#     assert tf.experimental.numpy.allclose(
#         gmm.gmm.components_distribution.scale_tril, scale_tril
#     )
#     assert tf.experimental.numpy.allclose(gmm.scale_tril, scale_tril)
#     assert tf.experimental.numpy.allclose(gmm.gmm.mixture_distribution.logits, log_w)
#     assert tf.experimental.numpy.allclose(gmm.log_w, log_w)

#     ## change GMM parameters
#     log_w = tf.math.log([0.2, 0.5, 0.3])
#     mu = tf.random.normal(shape=mu.shape)
#     scale_tril = tf.constant(
#         [
#             [[0.3, 0.0], [1.0, 0.8]],
#             [[4.0, 0.0], [-0.8, 2.0]],
#             [[1.0, 0.0], [-2.0, 1.0]],
#         ]
#     )
#     prec = tf.linalg.inv(
#         tf.matmul(scale_tril, tf.transpose(scale_tril, perm=(0, 2, 1)))
#     )
#     gmm.log_w = log_w
#     gmm.loc = mu
#     gmm.prec = prec

#     # check that setters work
#     assert tf.experimental.numpy.allclose(gmm.gmm.components_distribution.loc, mu)
#     assert tf.experimental.numpy.allclose(gmm.loc, mu)
#     assert tf.experimental.numpy.allclose(
#         gmm.gmm.components_distribution.scale_tril, scale_tril
#     )
#     assert tf.experimental.numpy.allclose(gmm.scale_tril, scale_tril)
#     assert tf.experimental.numpy.allclose(gmm.gmm.mixture_distribution.logits, log_w)
#     assert tf.experimental.numpy.allclose(gmm.log_w, log_w)


# def test_target_dist_wrapper():
#     # generate GMM, which we will use as target dist
#     w = np.array([0.1, 0.5, 0.4])
#     mu = np.array(
#         [
#             [-1.0, -1.0],
#             [-1.0, 1.0],
#             [1.0, 0.0],
#         ]
#     )
#     scale_tril = np.array(
#         [
#             [[0.1, 0.0], [0.0, 0.5]],
#             [[2.0, 0.0], [-0.5, 3.0]],
#             [[1.0, 0.0], [-4.0, 1.0]],
#         ]
#     )
#     cov = np.matmul(scale_tril, tf.transpose(scale_tril, perm=(0, 2, 1)))
#     K = w.shape[0]
#     D = mu.shape[1]
#     gmm = GMM_LNPDF(target_weights=w, target_means=mu, target_covars=cov)

#     # generate samples
#     S = 10
#     z = gmm.sample(S)

#     # generate target dist
#     target_dist = TargetDistWrapper(target_dist=gmm)
#     del gmm  # we only need target_dist from now on

#     ## test computation log_marginal_z using a test function with known grad/hess
#     log_density, log_density_grad, log_density_hess = target_dist.log_density_grad_hess(
#         z=z,
#         compute_grad=True,
#         compute_hess=True,
#         test_fun=lambda z: fun_with_known_grad_hess(z)[0],
#     )
#     assert log_density.shape == (S,)
#     assert log_density_grad.shape == (S, D)
#     assert log_density_hess.shape == (S, D, D)
#     (
#         true_log_density,
#         true_log_density_grad,
#         true_log_density_hess,
#     ) = fun_with_known_grad_hess(z)
#     # TODO:
#     # the values should match exactly, but for some runs they do and for others there
#     # are miniscule differences -> why?
#     assert tf.experimental.numpy.allclose(log_density, true_log_density)
#     assert tf.experimental.numpy.allclose(log_density_grad, true_log_density_grad)
#     assert tf.experimental.numpy.allclose(log_density_hess, true_log_density_hess)

#     ## test computation log_marginal_z using the GMM likelihood
#     log_density, log_density_grad, log_density_hess = target_dist.log_density_grad_hess(
#         z=z, compute_grad=True, compute_hess=True
#     )
#     assert log_density.shape == (S,)
#     assert log_density_grad.shape == (S, D)
#     assert log_density_hess.shape == (S, D, D)

#     log_density, log_density_grad, log_density_hess = target_dist.log_density_grad_hess(
#         z=z, compute_grad=True, compute_hess=False
#     )
#     assert log_density.shape == (S,)
#     assert log_density_grad.shape == (S, D)
#     assert log_density_hess is None

#     log_density, log_density_grad, log_density_hess = target_dist.log_density_grad_hess(
#         z=z, compute_grad=False, compute_hess=False
#     )
#     assert log_density.shape == (S,)
#     assert log_density_grad is None
#     assert log_density_hess is None
