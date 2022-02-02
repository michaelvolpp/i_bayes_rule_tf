import os
import time

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from i_bayes_rule.util import (
    GMM,
    TargetDistWrapper,
    eval_grad_hess,
    expectation_prod_neg,
    compute_S_bar,
    log_omega_to_log_w,
    log_w_to_log_omega,
    scale_tril_to_cov,
)


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
    assert tf.rank(w_init) == 1  # vector of scalar mixture weights
    K = w_init.shape[0]  # number of components
    # means
    assert tf.rank(mu_init) == 2
    assert mu_init.shape[0] == K
    D = mu_init.shape[1]
    # covariances
    assert cov_init.shape == (K, D, D)
    # savepath
    os.makedirs(config["savepath"], exist_ok=True)
    # check compatibility of model and target dist
    assert target_dist.get_num_dimensions() == D

    ## instantiate model and target distribution
    model = GMM(
        log_w=tf.math.log(w_init),
        loc=mu_init,
        prec=tf.linalg.inv(cov_init),  # TODO: do we get around this?
    )
    tgt_dist_wrap = TargetDistWrapper(target_dist=target_dist)

    ## training loop
    n_fevals_total = 0
    total_wall_time = 0
    with tqdm(total=config["n_iter"]) as pbar:
        for i in range(config["n_iter"]):
            # update parameters
            t1 = time.time()
            model, n_fevals = step(
                tgt_dist_wrap=tgt_dist_wrap,
                model=model,
                n_samples=config["n_samples_per_iter"],
                lr_w=config["lr_w"],
                lr_mu_prec=config["lr_mu_prec"],
                prec_method=config["prec_update_method"],
                lr_mu_prec_gamma=config["lr_mu_prec_gamma"],
                lr_w_gamma=config["lr_w_gamma"],
            )
            t2 = time.time()
            # save wall time every step
            wall_time = t2 - t1
            total_wall_time += wall_time
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
                tgt_ll, elbo = log_results(
                    savepath=config["savepath"],
                    model=model,
                    tgt_dist_wrap=tgt_dist_wrap,
                    n_fevals_total=n_fevals_total,
                    iteration=i,
                    pbar=pbar,
                    total_wall_time=total_wall_time,
                )
                if wandb_logger is not None:
                    wandb_logger.process(
                        {
                            "iter": i,
                            "n_fevals_total": n_fevals_total,
                            "target_ll": tgt_ll,
                            "-elbo": -elbo,
                        }
                    )

    return model


def log_results(
    savepath,
    model,
    tgt_dist_wrap,
    n_fevals_total,
    iteration,
    pbar,
    total_wall_time=None,
):
    # draw test samples and compute target log likelihood
    z = model.sample(2000)
    model_ll, _, _ = model.log_density_grad_hess(
        z=z, compute_grad=False, compute_hess=False
    )
    model_ll = tf.reduce_mean(model_ll)
    tgt_ll, _, _ = tgt_dist_wrap.log_density_grad_hess(
        z=z, compute_grad=False, compute_hess=False
    )
    tgt_ll = tf.reduce_mean(tgt_ll)
    elbo = tgt_ll - model_ll

    # convert results to numpy
    w = tf.exp(model.log_w).numpy()
    mu = model.loc.numpy()
    # TODO: is this more stable than directly computing tf.lingal.inv(model.prec)?
    cov = scale_tril_to_cov(model.scale_tril).numpy()
    tgt_ll = tgt_ll.numpy()
    elbo = elbo.numpy()

    pbar.set_postfix(
        {"n_evals": n_fevals_total, "avg. sample logpdf": tgt_ll, "ELBO": elbo}
    )

    # save GMM parameters to file
    os.makedirs(os.path.join(savepath, "gmm_dump"), exist_ok=True)
    np.savez(
        file=os.path.join(savepath, "gmm_dump", f"gmm_dump_{iteration:01d}.npz"),
        weights=w,
        means=mu,
        covs=cov,
        timestamps=time.time(),
        fevals=n_fevals_total,
    )
    if total_wall_time is not None:
        wall_time_path = os.path.join(savepath, "wall_time_dump")
        os.makedirs(wall_time_path, exist_ok=True)
        np.save(
            os.path.join(wall_time_path, f"total_wall_time_dump_{iteration:01d}.npy"),
            total_wall_time,
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
    # check input
    n_components = model.log_w.shape[-1]
    d_z = model.loc.shape[-1]
    batch_shape = model.loc.shape[:-1]

    ## sample z from GMM model
    z = model.sample(n_samples=n_samples)
    assert z.shape == (n_samples,) + batch_shape + (d_z,)

    ## evaluate target distribution at samples, i.e.,
    # the (unnormalized) log posterior density, and its gradient + hessian
    compute_hess = prec_method == "hessian"
    (
        log_tgt_density,
        log_tgt_density_grad,
        log_tgt_density_hess,  # we only require this for the hessian method
    ) = eval_grad_hess(
        fun=target_dist, z=z, compute_grad=True, compute_hess=compute_hess
    )
    assert log_tgt_density.shape == (n_samples,) + batch_shape
    assert log_tgt_density_grad.shape == (n_samples,) + batch_shape + (d_z,)
    if compute_hess:
        assert log_tgt_density_hess.shape == (n_samples,) + batch_shape + (d_z, d_z)
    n_feval = z.shape[0]  # TODO: is this correct?

    ## evaluate the GMM model, and its gradient + hessian at samples
    (
        log_model_density,
        log_model_density_grad,
        log_model_density_hess,
    ) = eval_grad_hess(fun=model.log_density, z=z, compute_grad=True, compute_hess=True)
    assert log_model_density.shape == (n_samples,) + batch_shape
    assert log_model_density_grad.shape == (n_samples,) + batch_shape + (d_z,)
    assert log_model_density_hess.shape == (n_samples,) + batch_shape + (d_z, d_z)

    ## compute delta(z)
    # TODO: they do this somehow differently
    log_delta_z = model.log_component_densities(z=z) - log_model_density[..., None]
    assert log_delta_z.shape == (n_samples,) + batch_shape + (n_components,)

    # TODO: learning rate decay?

    ## update model parameters
    new_log_w = update_log_w(
        log_w=model.log_w,
        log_delta_z=log_delta_z,
        log_tgt_density=log_tgt_density,
        log_model_density=log_model_density,
        lr=lr_w,
    )
    new_mu = update_mu(
        mu=model.loc,
        prec_tril=model.prec_tril,
        log_delta_z=log_delta_z,
        log_tgt_density_grad=log_tgt_density_grad,
        log_model_density_grad=log_model_density_grad,
        lr=lr_mu_prec,
    )
    if prec_method == "hessian":
        g = compute_g_hessian(
            log_delta_z=log_delta_z,
            log_tgt_post_hess=log_tgt_density_hess,
            log_model_marg_z_hess=log_model_density_hess,
        )
    else:
        assert prec_method == "reparam"
        g = compute_g_reparam(
            z=z,
            mu=model.loc,
            prec=model.prec,
            log_delta_z=log_delta_z,
            log_tgt_post_grad=log_tgt_density_grad,
            log_model_marg_z_hess=log_model_density_hess,
        )
    assert g.shape == batch_shape + (n_components, d_z, d_z)
    new_prec = update_prec(
        prec=model.prec,
        prec_tril=model.prec_tril,
        g=g,
        lr=lr_mu_prec,
    )

    # check output
    assert new_log_w.shape == model.log_w.shape
    assert new_mu.shape == model.loc.shape
    assert new_prec.shape == model.prec.shape
    model.log_w = new_log_w
    model.loc = new_mu
    model.prec = new_prec
    return model, n_feval


def update_log_w(
    log_w: tf.Tensor,
    log_delta_z: tf.Tensor,
    log_tgt_density: tf.Tensor,
    log_model_density: tf.Tensor,
    lr: float,
):
    # check inputs
    batch_shape = log_w.shape[:-1]
    n_components = log_w.shape[-1]
    n_samples = log_tgt_density.shape[0]
    assert log_w.shape == batch_shape + (n_components,)
    assert log_delta_z.shape == (n_samples,) + batch_shape + (n_components,)
    assert log_tgt_density.shape == (n_samples,) + batch_shape
    assert log_model_density.shape == (n_samples,) + batch_shape

    # computations are performed in log_omega space
    #  where omega[k] := w[k] / w[K] for k=1..K-1
    log_omega = log_w_to_log_omega(log_w)

    ## update log_omega
    b_z = -log_tgt_density + log_model_density
    assert b_z.shape == (n_samples,) + batch_shape
    # compute E_q[delta(z) * b(z)]
    expectation = expectation_prod_neg(log_a_z=log_delta_z, b_z=b_z[..., None])
    assert expectation.shape == batch_shape + (n_components,)
    # compute E_q[(delta(z)[:-1] - delta(z)[-1])*b(z)]
    d_log_omega = expectation[..., :-1] - expectation[..., -1]
    # update log_omega
    d_log_omega = -lr * d_log_omega
    log_omega = log_omega + d_log_omega

    # go back to log_w space
    log_w = log_omega_to_log_w(log_omega)

    # check outputs
    assert log_w.shape == batch_shape + (n_components,)
    return log_w


def update_mu(
    mu: tf.Tensor,
    prec_tril: tf.Tensor,
    log_delta_z: tf.Tensor,
    log_tgt_density_grad: tf.Tensor,
    log_model_density_grad: tf.Tensor,
    lr: float,
):
    # check inputs
    batch_shape = mu.shape[:-2]
    n_components = mu.shape[-2]
    d_z = mu.shape[-1]
    n_samples = log_tgt_density_grad.shape[0]
    assert mu.shape == batch_shape + (n_components, d_z)
    assert prec_tril.shape == batch_shape + (n_components, d_z, d_z)
    assert log_delta_z.shape == (n_samples,) + batch_shape + (n_components,)
    assert log_tgt_density_grad.shape == (n_samples,) + batch_shape + (d_z,)
    assert log_model_density_grad.shape == (n_samples,) + batch_shape + (d_z,)

    ## update mu
    b_z_grad = -log_tgt_density_grad + log_model_density_grad
    assert b_z_grad.shape == (n_samples,) + batch_shape + (d_z,)
    # compute E_q[delta(z) * d/dz(b(z))]
    expectation = expectation_prod_neg(
        log_a_z=log_delta_z[..., None], b_z=b_z_grad[..., None, :]
    )
    assert expectation.shape == batch_shape + (n_components, d_z)
    # compute S^{-1} * E_q[delta(z)*grad_z(b(z))]
    d_mu = tf.linalg.cholesky_solve(
        chol=prec_tril,
        rhs=expectation[..., None],
    )[..., 0]
    assert d_mu.shape == batch_shape + (n_components, d_z)
    # update mu
    d_mu = -lr * d_mu
    mu = mu + d_mu

    # check outputs
    assert mu.shape == batch_shape + (n_components, d_z)
    return mu


@tf.function
def compute_g_hessian(
    log_delta_z: tf.Tensor,
    log_tgt_density_hess: tf.Tensor,
    log_model_density_hess: tf.Tensor,
):
    # check inputs
    n_samples = log_delta_z.shape[0]
    n_components = log_delta_z.shape[-1]
    batch_shape = log_delta_z.shape[1:-1]
    d_z = log_tgt_density_hess.shape[-1]
    assert log_delta_z.shape == (n_samples,) + batch_shape + (n_components,)
    assert log_tgt_density_hess.shape == (n_samples,) + batch_shape + (d_z, d_z)
    assert log_model_density_hess.shape == (n_samples,) + batch_shape + (d_z, d_z)

    # compute g = -E_q[delta(z) * d^2/dz^2 b(z)]
    b_z_hess = -log_tgt_density_hess + log_model_density_hess
    assert b_z_hess.shape == (n_samples,) + batch_shape + (d_z, d_z)
    g = -expectation_prod_neg(
        log_a_z=log_delta_z[..., None, None], b_z=b_z_hess[..., None, :, :]
    )

    # check output
    assert g.shape == batch_shape + (n_components, d_z, d_z)
    return g


@tf.function
def compute_g_reparam(
    z: tf.Tensor,
    mu: tf.Tensor,
    prec: tf.Tensor,
    log_delta_z: tf.Tensor,
    log_tgt_density_grad: tf.Tensor,
    log_model_density_hess: tf.Tensor,
):
    # check inputs
    n_samples = log_delta_z.shape[0]
    n_components = log_delta_z.shape[-1]
    batch_shape = log_delta_z.shape[1:-1]
    d_z = log_tgt_density_grad.shape[-1]
    assert z.shape == n_samples + batch_shape + (d_z,)
    assert mu.shape == batch_shape + (n_components, d_z)
    assert prec.shape == batch_shape + (n_components, d_z, d_z)
    assert log_delta_z.shape == (n_samples,) + batch_shape + (n_components,)
    assert log_tgt_density_grad.shape == (n_samples,) + batch_shape + (d_z, d_z)
    assert log_model_density_hess.shape == (n_samples,) + batch_shape + (d_z, d_z)

    ## compute g = -E_q[delta(z) * ((S_bar + S_bar^T)/2 + d^2/dz^2 log(q(z))]
    S_bar = compute_S_bar(prec=prec, z=z, mu=mu, log_tgt_post_grad=log_tgt_density_grad)
    # symmetrize
    S_bar_sym = 0.5 * (S_bar + tf.linalg.matrix_transpose(S_bar))
    assert S_bar_sym.shape == (n_samples,) + batch_shape + (n_components, d_z, d_z)
    # compute g
    b_z_reparam = S_bar_sym + log_model_density_hess[..., None, :, :]
    assert b_z_reparam.shape == (n_samples,) + batch_shape + (n_components, d_z, d_z)
    g = -expectation_prod_neg(log_a_z=log_delta_z[..., None, None], b_z=b_z_reparam)

    # check output
    assert g.shape == batch_shape + (n_components, d_z, d_z)
    return g


@tf.function
def update_prec(prec: tf.Tensor, prec_tril: tf.Tensor, g: tf.Tensor, lr: float):
    # check input
    batch_shape = prec.shape[:-3]
    n_components = prec.shape[-3]
    d_z = prec.shape[-1]
    assert prec.shape == batch_shape + (n_components, d_z, d_z)
    assert prec_tril.shape == batch_shape + (n_components, d_z, d_z)
    assert g.shape == batch_shape + (n_components, d_z, d_z)

    ## update precision
    # solve linear equations
    sols = tf.linalg.cholesky_solve(chol=prec_tril, rhs=g)
    assert sols.shape == batch_shape + (n_components, d_z, d_z)
    # compute update
    d_prec = -lr * g
    d_prec = d_prec + 0.5 * lr ** 2 * tf.einsum("...kij,...kjl->...kil", g, sols)
    assert d_prec.shape == batch_shape + (n_components, d_z, d_z)
    prec = prec + d_prec

    # TODO: the following might be more stable?
    # U = tf.transpose(prec_tril, [0, 2, 1]) - lr * tf.linalg.cholesky_solve(
    #     chol=prec_tril, rhs=g
    # )
    # prec = 0.5 * (prec + tf.transpose(U, [0, 2, 1]) @ U)
    # prec = 0.5 * (prec + tf.transpose(prec, [0, 2, 1]))

    # check output
    assert prec.shape == batch_shape + (n_components, d_z, d_z)
    return prec
