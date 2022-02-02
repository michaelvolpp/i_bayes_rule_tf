import os
import time

import numpy as np
import tensorflow as tf
import wandb
from tqdm import tqdm
from cw2.cw_data.cw_wandb_logger import WandBLogger
import tensorflow_probability as tfp

from i_bayes_rule.util import (
    GMM,
    TargetDistWrapper,
    expectation_prod_neg,
    compute_S_bar,
    log_omega_to_log_w,
    log_w_to_log_omega,
    scale_tril_to_cov
)
from target_lnpdfs.Lnpdf import LNPDF


def create_initial_model(D, K, prior_scale, initial_cov=None):
    # Source: Oleg
    if np.isscalar(prior_scale):
        prior = tfp.distributions.MultivariateNormalDiag(loc=np.zeros(D), scale_identity_multiplier=prior_scale)
    else:
        prior = tfp.distributions.MultivariateNormalDiag(loc=np.zeros(D), scale_diag=prior_scale)

    if initial_cov is None:
        initial_cov = prior.covariance().numpy().astype(np.float32) # use the same initial covariance that was used for sampling the mean
    else:
        if np.isscalar(initial_cov):
            initial_cov = initial_cov * tf.eye(D)

    weights = np.ones(K) / K
    means = np.zeros((K, D))
    covs = np.zeros((K, D, D))
    for i in range(0, K):
        if K == 1:
            means[i] = np.zeros(D)
        else:
            means[i] = prior.sample(1).numpy()
        # use the same initial covariance that was used for sampling the mean
        covs[i] = initial_cov

    return weights, means, covs

def i_bayesian_learning_rule_gmm_grid_search(
    config: dict,
    target_dist: LNPDF,
    w_init: tf.Tensor,
    mu_init: tf.Tensor,
    cov_init: tf.Tensor,
    callback,
    wandb_logger: WandBLogger = None
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
    failed = False
    with tqdm(total=config["n_iter"]) as pbar:
        for i in range(config["n_iter"]):
            # update parameters
            model, n_fevals = step(
                tgt_dist_wrap=tgt_dist_wrap,
                model=model,
                n_samples=config["n_samples_per_iter"],
                lr_w=config["lr_w"],
                lr_mu_prec=config["lr_mu_prec"],
                prec_method=config["prec_update_method"],
                lr_mu_prec_gamma=config["lr_mu_prec_gamma"],
                lr_w_gamma=config["lr_w_gamma"]
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
                tgt_ll, elbo = log_results(
                    savepath=config["savepath"],
                    model=model,
                    tgt_dist_wrap=tgt_dist_wrap,
                    n_fevals_total=n_fevals_total,
                    iteration=i,
                    pbar=pbar,
                )
                if np.isnan(elbo):
                    print("ELBO is none")
                    failed = True
                    break
                if wandb_logger is not None:
                    wandb_logger.process({"iter": i, "n_fevals_total": n_fevals_total,
                                          "target_ll": tgt_ll, "-elbo": -elbo})
            if n_fevals_total > config["max_samples"]:
                print("max samples passed!")
                break
    if wandb_logger is not None:
        if failed:
            wandb.log({"NAN": 1})
        else:
            wandb.log({"NAN": 0})
    return model



def i_bayesian_learning_rule_gmm(
    config: dict,
    target_dist: LNPDF,
    w_init: tf.Tensor,
    mu_init: tf.Tensor,
    cov_init: tf.Tensor,
    callback,
    wandb_logger: WandBLogger = None
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
                lr_w_gamma=config["lr_w_gamma"]
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
                    wandb_logger.process({"iter": i, "n_fevals_total": n_fevals_total,
                                          "target_ll": tgt_ll, "-elbo": -elbo})

    return model


def log_results(savepath, model, tgt_dist_wrap, n_fevals_total, iteration, pbar, total_wall_time=None):
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
            os.path.join(wall_time_path, f'total_wall_time_dump_{iteration:01d}.npy'),
            total_wall_time
        )
    return tgt_ll, elbo


def step(tgt_dist_wrap, model, n_samples, lr_w, lr_mu_prec, prec_method, lr_mu_prec_gamma, lr_w_gamma):
    step.t = step.t + 1
    ## check input
#    K = model.log_w.shape[0]
#    D = model.loc.shape[1]

    ## sample z from GMM model
    z = model.sample(n_samples=n_samples)
   # assert z.shape == (n_samples, D)
    n_feval = z.shape[0]  # TODO: is this correct?

    ## evaluate target distribution at samples, i.e.,
    # the (unnormalized) log posterior density, and its gradient + hessian
    compute_hess = prec_method == "hessian"
    (
        log_tgt_post,
        log_tgt_post_grad,
        log_tgt_post_hess,  # we only require this for the hessian method
    ) = tgt_dist_wrap.log_density_grad_hess(
        z=z, compute_grad=True, compute_hess=compute_hess
    )
 #   assert log_tgt_post.shape == (n_samples,)
  #  assert log_tgt_post_grad.shape == (n_samples, D)
  #  if compute_hess:
  #      assert log_tgt_post_hess.shape == (n_samples, D, D)

    ## evaluate the GMM model, and its gradient + hessian at samples
    (
        log_model_marg_z,
        log_model_marg_z_grad,
        log_model_marg_z_hess,
    ) = model.log_density_grad_hess(z=z, compute_grad=True, compute_hess=True)
  #  assert log_model_marg_z.shape == (n_samples,)
  #  assert log_model_marg_z_grad.shape == (n_samples, D)
  #  assert log_model_marg_z_hess.shape == (n_samples, D, D)

    ## compute delta(z)
    # -> TODO: they do this somehow differently
    log_delta_z = model.log_component_densities(z=z) - log_model_marg_z[:, None]
 #   assert log_delta_z.shape == (n_samples, K)

    # lr_w_gamma and lr_mu_prec_gamma specify decay rates for the stepsize according to
    # Eq (104) in Khan, et al. "Fast and scalable bayesian deep learning by weight-perturbation in adam." ICML 2018.
    # for negative decays we keep the corresponding stepsize constant
    if lr_w_gamma > 0:
        lr_w_t = lr_w / (1+step.t ** lr_w_gamma)
    else:
        lr_w_t = lr_w
    if lr_mu_prec_gamma > 0:
        lr_mu_prec_t = lr_mu_prec / (1+step.t ** lr_mu_prec_gamma)
    else:
        lr_mu_prec_t = lr_mu_prec


    ## update model parameters
    new_log_w = update_log_w(
        log_w=model.log_w,
        log_delta_z=log_delta_z,
        log_tgt_post=log_tgt_post,
        log_model_marg_z=log_model_marg_z,
        lr=lr_w_t,
    )
    new_mu = update_mu(
        mu=model.loc,
        prec_tril=model.prec_tril,
        log_delta_z=log_delta_z,
        log_tgt_post_grad=log_tgt_post_grad,
        log_model_marg_z_grad=log_model_marg_z_grad,
        lr=lr_mu_prec_t,
    )
    if prec_method == "hessian":
        g = compute_g_hessian(
            log_delta_z=log_delta_z,
            log_tgt_post_hess=log_tgt_post_hess,
            log_model_marg_z_hess=log_model_marg_z_hess,
        )
    else:
#        assert prec_method == "reparam"
        g = compute_g_reparam(
            z=z,
            mu=model.loc,
            prec=model.prec,
            log_delta_z=log_delta_z,
            log_tgt_post_grad=log_tgt_post_grad,
            log_model_marg_z_hess=log_model_marg_z_hess,
        )
#    assert g.shape == (K, D, D)
    new_prec = update_prec(
        prec=model.prec,
        prec_tril=model.prec_tril,
        g=g,
        lr=lr_mu_prec,
    )

    # check output
    model.log_w = new_log_w
    model.loc = new_mu
    model.prec = new_prec
    return model, n_feval
step.t = 0 # initialize step counter

def update_log_w(log_w, log_delta_z, log_tgt_post, log_model_marg_z, lr):
    # check inputs
   # K = log_w.shape[0]

    # computations are performed in log_omega space
    #  where omega[k] := w[k] / w[K] for k=1..K-1
    log_omega = log_w_to_log_omega(log_w)

    ## update log_omega
    b_z = -log_tgt_post + log_model_marg_z
    # compute E_q[delta(z) * b(z)]
    expectation = expectation_prod_neg(log_a_z=log_delta_z, b_z=b_z[:, None])
  #  assert expectation.shape == (K,)
    # compute E_q[(delta(z)[:-1] - delta(z)[-1])*b(z)]
    d_log_omega = expectation[:-1] - expectation[-1]
    # update log_omega
    d_log_omega = -lr * d_log_omega
    log_omega = log_omega + d_log_omega

    # go back to log_w space
    log_w = log_omega_to_log_w(log_omega)

    # check outputs
  #  assert log_w.shape == (K,)
    return log_w


def update_mu(mu, prec_tril, log_delta_z, log_tgt_post_grad, log_model_marg_z_grad, lr):
    # check inputs
 #   K = mu.shape[0]
 #   D = mu.shape[1]

    ## update mu
    b_z_grad = -log_tgt_post_grad + log_model_marg_z_grad
    # compute E_q[delta(z) * d/dz(b(z))]
    expectation = expectation_prod_neg(
        log_a_z=log_delta_z[:, :, None], b_z=b_z_grad[:, None, :]
    )
    # expectation_k = np.exp(log_delta_z + np.log(b_z_grad))
    # expectation_k = np.mean(expectation_k, axis=0)
 #   assert expectation.shape == (K, D)
    # compute S^{-1} * E_q[delta(z)*grad_z(b(z))]
    d_mu = tf.linalg.cholesky_solve(
        chol=prec_tril,
        rhs=expectation[:, :, None],
    )[:, :, 0]
  #  assert d_mu.shape == (K, D)
    # update mu
    d_mu = -lr * d_mu
    mu = mu + d_mu

    # check outputs
 #   assert mu.shape == (K, D)
    return mu

@tf.function
def compute_g_hessian(
    log_delta_z,
    log_tgt_post_hess,
    log_model_marg_z_hess,
):
    # check inputs
    S = log_delta_z.shape[0]
    K = log_delta_z.shape[1]
    D = log_tgt_post_hess.shape[1]
    assert log_delta_z.shape == (S, K)
    assert log_tgt_post_hess.shape == (S, D, D)
    assert log_model_marg_z_hess.shape == (S, D, D)

    # compute g = -E_q[delta(z) * d^2/dz^2 b(z)]
    b_z_hess = -log_tgt_post_hess + log_model_marg_z_hess
    g = -expectation_prod_neg(
        log_a_z=log_delta_z[:, :, None, None], b_z=b_z_hess[:, None, :, :]
    )
    # g = np.exp(log_delta_z + np.log(b_z_hess))
    # g = -np.mean(g, axis=0)

    # check output
    assert g.shape == (K, D, D)
    return g


@tf.function
def compute_g_reparam(
    z,
    mu,
    prec,
    log_delta_z,
    log_tgt_post_grad,
    log_model_marg_z_hess,
):
    # check inputs
 #   S = z.shape[0]
 #   K = mu.shape[0]
 #   D = mu.shape[1]
  #  assert z.shape == (S, D)
  #  assert mu.shape == (K, D)
   # assert prec.shape == (K, D, D)
    #assert log_tgt_post_grad.shape == (S, D)
   # assert log_delta_z.shape == (S, K)
   # assert log_model_marg_z_hess.shape == (S, D, D)

    ## compute g = -E_q[delta(z) * ((S_bar + S_bar^T)/2 + d^2/dz^2 log(q(z))]
    S_bar = compute_S_bar(prec=prec, z=z, mu=mu, log_tgt_post_grad=log_tgt_post_grad)
    # symmetrize
    S_bar_sym = 0.5 * (S_bar + tf.transpose(S_bar, perm=(0, 1, 3, 2)))
    # compute g
    b_z_reparam = S_bar_sym + log_model_marg_z_hess[:, None, :, :]
    g = -expectation_prod_neg(log_a_z=log_delta_z[:, :, None, None], b_z=b_z_reparam)
    # g = np.exp(log_delta_z + np.log(S_bar_sym + log_model_marg_z_hess))
    # g = -np.mean(g, axis=0)

    # check output
  #  assert g.shape == (K, D, D)
    return g

@tf.function
def update_prec(prec, prec_tril, g, lr):
    # check input
  #  K = prec.shape[0]
  #  D = prec.shape[1]
  #  assert prec.shape == (K, D, D)
  #  assert prec_tril.shape == (K, D, D)
  #  assert g.shape == (K, D, D)

    ## update precision
    # solve linear equations
    if True:
        sols = tf.linalg.cholesky_solve(chol=prec_tril, rhs=g)
    #    assert sols.shape == (K, D, D)
        # compute update
        d_prec = -lr * g
        d_prec = d_prec + 0.5 * lr ** 2 * tf.einsum("kij,kjl->kil", g, sols)
     #   assert d_prec.shape == (K, D, D)
        prec = prec + d_prec
    else:
        U = tf.transpose(prec_tril, [0, 2, 1]) - lr * tf.linalg.cholesky_solve(chol=prec_tril, rhs=g)
        prec = 0.5 * (prec + tf.transpose(U, [0, 2, 1]) @ U)
        prec = 0.5 * (prec + tf.transpose(prec, [0, 2, 1]))

    # check output
  #  assert prec.shape == (K, D, D)
    return prec
