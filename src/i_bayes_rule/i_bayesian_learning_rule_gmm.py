import os
import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from i_bayes_rule.lnpdf import LNPDF
from i_bayes_rule.util import (
    compute_S_bar,
    expectation_prod_neg,
    log_omega_to_log_w,
    log_w_to_log_omega,
)
from gmm_util.gmm import GMM
from gmm_util.util import scale_tril_to_cov, assert_shape


def i_bayesian_learning_rule_gmm(
    config: dict,
    target_dist: LNPDF,
    w_init: tf.Tensor,
    mu_init: tf.Tensor,
    cov_init: tf.Tensor,
    callback,
):
    ## check inputs
    # mixture weights
    batch_shape = tf.shape(w_init)[:-1]
    n_components = tf.shape(w_init)[-1]
    d_z = tf.shape(mu_init)[-1]
    assert_shape(w_init, (batch_shape, n_components))
    assert_shape(mu_init, (batch_shape, n_components, d_z))
    assert_shape(cov_init, (batch_shape, n_components, d_z, d_z))

    # savepath
    os.makedirs(config["savepath"], exist_ok=True)
    # check compatibility of model and target dist
    assert target_dist.get_num_dimensions() == d_z

    ## instantiate model and target distribution
    model = GMM(
        log_w=tf.math.log(w_init),
        loc=mu_init,
        prec=tf.linalg.inv(cov_init),  # TODO: do we get around this?
    )

    ## training loop
    n_fevals_total = 0
    with tqdm(total=config["n_iter"]) as pbar:
        for i in range(config["n_iter"]):
            # update parameters
            model, n_fevals, _ = step(
                target_dist=target_dist,
                model=model,
                n_samples=config["n_samples_per_iter"],
                lr_w=config["lr_w"],
                lr_mu_prec=config["lr_mu_prec"],
                prec_method=config["prec_update_method"],
            )
            # update n_fevals_total
            # TODO: what does n_fevals actually count
            n_fevals_total += n_fevals

            # log
            pbar.update()
            if callback is not None and (
                i % config["callback_interval"] == 0 or i == config["n_iter"] - 1
            ):
                callback(model=model)
            if i % config["log_interval"] == 0 or i == config["n_iter"] - 1:
                log_results(
                    model=model,
                    target_dist=target_dist,
                    savepath=config["savepath"],
                    n_fevals_total=n_fevals_total,
                    iteration=i,
                    pbar=pbar,
                )

    return model


def log_results(
    model: GMM,
    target_dist: LNPDF,
    savepath: str,
    n_fevals_total: int,
    iteration: int,
    pbar,
    save_to_file=False,
):
    # draw test samples and compute target log likelihood
    z = model.sample(2000)
    model_ll, _, _ = model.log_density(z=z)
    model_ll = tf.reduce_mean(model_ll)
    tgt_ll, _, _ = target_dist.log_density(z=z)
    tgt_ll = tf.reduce_mean(tgt_ll)
    elbo = tgt_ll - model_ll

    # convert results to numpy
    w = tf.exp(model.log_w).numpy()
    mu = model.loc.numpy()
    # TODO: is this more stable than directly computing tf.lingal.inv(model.prec)?
    cov = scale_tril_to_cov(model.scale_tril).numpy()
    tgt_ll = tgt_ll.numpy()
    elbo = elbo.numpy()

    # update pbar
    pbar.set_postfix(
        {"n_evals": n_fevals_total, "avg. sample logpdf": tgt_ll, "ELBO": elbo}
    )

    # save GMM parameters to file
    if save_to_file:
        os.makedirs(os.path.join(savepath, "gmm_dump"), exist_ok=True)
        np.savez(
            file=os.path.join(savepath, "gmm_dump", f"gmm_dump_{iteration:01d}.npz"),
            weights=w,
            means=mu,
            covs=cov,
            timestamps=time.time(),
            fevals=n_fevals_total,
        )

    return tgt_ll, elbo


def step(
    model,
    target_dist,
    n_samples,
    lr_w,
    lr_mu_prec,
    prec_method,
):
    assert tf.executing_eagerly()

    # check input
    n_components = tf.shape(model.log_w)[-1]
    d_z = tf.shape(model.loc)[-1]
    batch_shape = tf.shape(model.loc)[:-2]

    ## sample z from GMM model
    z = model.sample(n_samples=n_samples)
    assert_shape(z, (n_samples, batch_shape, d_z))

    ## evaluate target distribution at samples, i.e.,
    # the (unnormalized) log posterior density, and its gradient + hessian
    compute_hess = prec_method == "hessian"
    (
        log_tgt_density,
        log_tgt_density_grad,
        log_tgt_density_hess,  # we only require this for the hessian method
    ) = target_dist.log_density(z=z, compute_grad=True, compute_hess=compute_hess)
    assert_shape(log_tgt_density, (n_samples, batch_shape))
    assert_shape(log_tgt_density_grad, (n_samples, batch_shape, d_z))
    if compute_hess:
        assert_shape(log_tgt_density_hess, (n_samples, batch_shape, d_z, d_z))
    n_feval = np.product(tf.shape(z)[:-1])  # TODO: is this correct?

    ## evaluate the GMM model, and its gradient + hessian at samples
    (
        log_model_density,
        log_model_density_grad,
        log_model_density_hess,
    ) = model.log_density(z=z, compute_grad=True, compute_hess=True)
    assert_shape(log_model_density, (n_samples, batch_shape))
    assert_shape(log_model_density_grad, (n_samples, batch_shape, d_z))
    assert_shape(log_model_density_hess, (n_samples, batch_shape, d_z, d_z))

    ## compute delta(z)
    # TODO: they do this somehow differently
    log_delta_z = model.log_component_densities(z=z) - log_model_density[..., None]
    assert_shape(log_delta_z, (n_samples, batch_shape, n_components))

    # TODO: learning rate decay?

    ## update model parameters
    new_log_w = update_log_w(
        log_w=model.log_w,
        log_delta_z=log_delta_z,
        log_target_density=log_tgt_density,
        log_model_density=log_model_density,
        lr=lr_w,
    )
    new_mu = update_mu(
        mu=model.loc,
        prec_tril=model.prec_tril,
        log_delta_z=log_delta_z,
        log_target_density_grad=log_tgt_density_grad,
        log_model_density_grad=log_model_density_grad,
        lr=lr_mu_prec,
    )
    if prec_method == "hessian":
        g = compute_g_hessian(
            log_delta_z=log_delta_z,
            log_target_density_hess=log_tgt_density_hess,
            log_model_density_hess=log_model_density_hess,
        )
    else:
        assert prec_method == "reparam"
        g = compute_g_reparam(
            z=z,
            mu=model.loc,
            prec=model.prec,
            log_delta_z=log_delta_z,
            log_target_density_grad=log_tgt_density_grad,
            log_model_density_hess=log_model_density_hess,
        )
    assert_shape(g, (batch_shape, n_components, d_z, d_z))
    new_prec = update_prec(
        prec=model.prec,
        prec_tril=model.prec_tril,
        g=g,
        lr=lr_mu_prec,
    )

    # check output
    assert_shape(new_log_w, tf.shape(model.log_w))
    assert_shape(new_mu, tf.shape(model.loc))
    assert_shape(new_prec, tf.shape(model.prec))
    model.log_w = new_log_w
    model.loc = new_mu
    model.prec = new_prec
    return model, n_feval, np.mean(log_tgt_density.numpy())


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),
        tf.TensorSpec(shape=None, dtype=tf.float32),
    ]
)
def update_log_w(
    log_w: tf.Tensor,
    log_delta_z: tf.Tensor,
    log_target_density: tf.Tensor,
    log_model_density: tf.Tensor,
    lr: float,
):
    # check inputs
    batch_shape = tf.shape(log_w)[:-1]
    n_components = tf.shape(log_w)[-1]
    n_samples = tf.shape(log_target_density)[0]
    assert_shape(log_w, (batch_shape, n_components))
    assert_shape(log_delta_z, (n_samples, batch_shape, n_components))
    assert_shape(log_target_density, (n_samples, batch_shape))
    assert_shape(log_model_density, (n_samples, batch_shape))

    # computations are performed in log_omega space
    #  where omega[k] := w[k] / w[K] for k=1..K-1
    log_omega = log_w_to_log_omega(log_w)

    ## update log_omega
    b_z = -log_target_density + log_model_density
    assert_shape(b_z, (n_samples, batch_shape))
    # compute E_q[delta(z) * b(z)]
    expectation = expectation_prod_neg(log_a_z=log_delta_z, b_z=b_z[..., None])
    assert_shape(expectation, (batch_shape, n_components))
    # compute E_q[(delta(z)[:-1] - delta(z)[-1])*b(z)]
    d_log_omega = expectation[..., :-1] - expectation[..., -1:]
    # update log_omega
    d_log_omega = -lr * d_log_omega
    log_omega = log_omega + d_log_omega

    # go back to log_w space
    log_w = log_omega_to_log_w(log_omega)

    # check outputs
    assert_shape(log_w, (batch_shape, n_components))
    return log_w


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=None, dtype=tf.float32),
    ]
)
def update_mu(
    mu: tf.Tensor,
    prec_tril: tf.Tensor,
    log_delta_z: tf.Tensor,
    log_target_density_grad: tf.Tensor,
    log_model_density_grad: tf.Tensor,
    lr: float,
):
    # check inputs
    batch_shape = tf.shape(mu)[:-2]
    n_components = tf.shape(mu)[-2]
    d_z = tf.shape(mu)[-1]
    n_samples = tf.shape(log_target_density_grad)[0]
    assert_shape(mu, (batch_shape, n_components, d_z))
    assert_shape(prec_tril, (batch_shape, n_components, d_z, d_z))
    assert_shape(log_delta_z, (n_samples, batch_shape, n_components))
    assert_shape(log_target_density_grad, (n_samples, batch_shape, d_z))
    assert_shape(log_model_density_grad, (n_samples, batch_shape, d_z))

    ## update mu
    b_z_grad = -log_target_density_grad + log_model_density_grad
    assert_shape(b_z_grad, (n_samples, batch_shape, d_z))
    # compute E_q[delta(z) * d/dz(b(z))]
    expectation = expectation_prod_neg(
        log_a_z=log_delta_z[..., None], b_z=b_z_grad[..., None, :]
    )
    assert_shape(expectation, (batch_shape, n_components, d_z))
    # compute S^{-1} * E_q[delta(z)*grad_z(b(z))]
    d_mu = tf.linalg.cholesky_solve(
        chol=prec_tril,
        rhs=expectation[..., None],
    )[..., 0]
    assert_shape(d_mu, (batch_shape, n_components, d_z))
    # update mu
    d_mu = -lr * d_mu
    mu = mu + d_mu

    # check outputs
    assert_shape(mu, (batch_shape, n_components, d_z))
    return mu


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
    ]
)
def compute_g_hessian(
    log_delta_z: tf.Tensor,
    log_target_density_hess: tf.Tensor,
    log_model_density_hess: tf.Tensor,
):
    # check inputs
    n_samples = tf.shape(log_delta_z)[0]
    n_components = tf.shape(log_delta_z)[-1]
    batch_shape = tf.shape(log_delta_z)[1:-1]
    d_z = tf.shape(log_target_density_hess)[-1]
    assert_shape(log_delta_z, (n_samples, batch_shape, n_components))
    assert_shape(log_target_density_hess, (n_samples, batch_shape, d_z, d_z))
    assert_shape(log_model_density_hess, (n_samples, batch_shape, d_z, d_z))

    # compute g = -E_q[delta(z) * d^2/dz^2 b(z)]
    b_z_hess = -log_target_density_hess + log_model_density_hess
    assert_shape(b_z_hess, (n_samples, batch_shape, d_z, d_z))
    g = -expectation_prod_neg(
        log_a_z=log_delta_z[..., None, None], b_z=b_z_hess[..., None, :, :]
    )

    # check output
    assert_shape(g, (batch_shape, n_components, d_z, d_z))
    return g


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
    ]
)
def compute_g_reparam(
    z: tf.Tensor,
    mu: tf.Tensor,
    prec: tf.Tensor,
    log_delta_z: tf.Tensor,
    log_target_density_grad: tf.Tensor,
    log_model_density_hess: tf.Tensor,
):
    # check inputs
    n_samples = tf.shape(log_delta_z)[0]
    n_components = tf.shape(log_delta_z)[-1]
    batch_shape = tf.shape(log_delta_z)[1:-1]
    d_z = tf.shape(log_target_density_grad)[-1]
    assert_shape(z, (n_samples, batch_shape, d_z))
    assert_shape(mu, (batch_shape, n_components, d_z))
    assert_shape(prec, (batch_shape, n_components, d_z, d_z))
    assert_shape(log_delta_z, (n_samples, batch_shape, n_components))
    assert_shape(log_target_density_grad, (n_samples, batch_shape, d_z))
    assert_shape(log_model_density_hess, (n_samples, batch_shape, d_z, d_z))

    ## compute g = -E_q[delta(z) * ((S_bar + S_bar^T)/2 + d^2/dz^2 log(q(z))]
    S_bar = compute_S_bar(
        prec=prec, z=z, loc=mu, log_tgt_post_grad=log_target_density_grad
    )
    # symmetrize
    S_bar_sym = 0.5 * (S_bar + tf.linalg.matrix_transpose(S_bar))
    assert_shape(S_bar_sym, (n_samples, batch_shape, n_components, d_z, d_z))
    # compute g
    b_z_reparam = S_bar_sym + log_model_density_hess[..., None, :, :]
    assert_shape(b_z_reparam, (n_samples, batch_shape, n_components, d_z, d_z))
    g = -expectation_prod_neg(log_a_z=log_delta_z[..., None, None], b_z=b_z_reparam)

    # check output
    assert_shape(g, (batch_shape, n_components, d_z, d_z))
    return g


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=None, dtype=tf.float32),
    ]
)
def update_prec(
    prec: tf.Tensor,
    prec_tril: tf.Tensor,
    g: tf.Tensor,
    lr: float,
):
    # check input
    batch_shape = tf.shape(prec)[:-3]
    n_components = tf.shape(prec)[-3]
    d_z = tf.shape(prec)[-1]
    assert_shape(prec, (batch_shape, n_components, d_z, d_z))
    assert_shape(prec_tril, (batch_shape, n_components, d_z, d_z))
    assert_shape(g, (batch_shape, n_components, d_z, d_z))

    ## update precision
    # solve linear equations
    sols = tf.linalg.cholesky_solve(chol=prec_tril, rhs=g)
    assert_shape(sols, (batch_shape, n_components, d_z, d_z))
    # compute update
    d_prec = -lr * g
    d_prec = d_prec + 0.5 * lr**2 * tf.einsum("...kij,...kjl->...kil", g, sols)
    assert_shape(d_prec, (batch_shape, n_components, d_z, d_z))
    prec = prec + d_prec

    # TODO: the following might be more stable?
    # U = tf.transpose(prec_tril, [0, 2, 1]) - lr * tf.linalg.cholesky_solve(
    #     chol=prec_tril, rhs=g
    # )
    # prec = 0.5 * (prec + tf.transpose(U, [0, 2, 1]) @ U)
    # prec = 0.5 * (prec + tf.transpose(prec, [0, 2, 1]))

    # check output
    assert_shape(prec, (batch_shape, n_components, d_z, d_z))
    return prec
