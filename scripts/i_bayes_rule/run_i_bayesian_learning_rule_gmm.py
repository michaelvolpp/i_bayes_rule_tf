import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from i_bayes_rule.i_bayesian_learning_rule_gmm import i_bayesian_learning_rule_gmm
from i_bayes_rule.lnpdf import make_simple_target, make_star_target, make_target


def create_initial_model(D, K, prior_scale, initial_cov=None):
    if np.isscalar(prior_scale):
        prior = tfp.distributions.MultivariateNormalDiag(
            loc=np.zeros(D), scale_identity_multiplier=prior_scale
        )
    else:
        prior = tfp.distributions.MultivariateNormalDiag(
            loc=np.zeros(D), scale_diag=prior_scale
        )

    if initial_cov is None:
        initial_cov = (
            prior.covariance().numpy().astype(np.float32)
        )  # use the same initial covariance that was used for sampling the mean
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


def plot2d(target_dist, model, fig, axes, block=False):
    def plot_gaussian_ellipse(ax, mean, scale_tril, color):
        n_plot = 100
        evals, evecs = np.linalg.eig(scale_tril @ scale_tril.T)
        theta = np.linspace(0, 2 * np.pi, n_plot)
        ellipsis = (np.sqrt(evals[None, :]) * evecs) @ [np.sin(theta), np.cos(theta)]
        ellipsis = ellipsis + mean[:, None]
        ax.plot(ellipsis[0, :], ellipsis[1, :], color=color)

    # create meshgrid
    n_plt = 25
    x = np.linspace(-5.0, 5.0, n_plt)
    y = np.linspace(-5.0, 5.0, n_plt)
    xx, yy = np.meshgrid(x, y)
    xy = np.vstack([xx.ravel(), yy.ravel()]).T

    # evaluate distributions
    xy_tf = tf.convert_to_tensor(xy, dtype=np.float32)
    p_tgt = np.exp(target_dist.log_density(xy_tf).numpy().reshape(n_plt, n_plt))
    p_model = np.exp(model.log_density(xy_tf).numpy().reshape(n_plt, n_plt))
    weights = np.exp(model.log_w.numpy())

    # plot target distribution
    ax = axes[0, 0]
    ax.clear()
    ax.contourf(xx, yy, p_tgt, levels=100)
    ax.axis("scaled")
    ax.set_title("Target density")
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")

    # plot model distribution
    ax = axes[1, 0]
    ax.clear()
    ax.contourf(xx, yy, p_model, levels=100)
    colors = []
    for k in range(model.n_components):
        color = next(ax._get_lines.prop_cycler)["color"]
        colors.append(color)
        scale_tril = model.scale_tril[k].numpy()
        mean = model.loc[k].numpy()
        ax.scatter(x=mean[0], y=mean[1])
        plot_gaussian_ellipse(ax=ax, mean=mean, scale_tril=scale_tril, color=color)
    ax.axis("scaled")
    ax.set_title("Model density")
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")

    # plot weights
    ax = axes[2, 0]
    ax.clear()
    ax.pie(weights, labels=[f"{w*100:.2f}%" for w in weights], colors=colors)
    ax.axis("scaled")
    ax.set_title("Mixture weights")

    fig.tight_layout()

    plt.show(block=block)
    plt.pause(0.001)


def main():
    # tf.config.run_functions_eagerly(True) # only for debugging

    ## create config
    scriptpath = os.path.dirname(os.path.abspath(__file__))
    config = dict()
    config["seed"] = 123
    config["log_interval"] = 500  # how often the parameters are written to npz files
    config["callback_interval"] = 500  # how often the callback is called
    config["n_dimensions_model"] = 2
    # config["prec_update_method"] = "reparam"
    config["prec_update_method"] = "hessian"
    # The following are the standard parameters from Lin et al. (for the GMM-star exp)
    config["n_iter"] = int(1e4)
    config["n_samples_per_iter"] = 50
    config["lr_mu_prec"] = 0.01
    config["lr_mu_prec_gamma"] = -1.0
    config["lr_w"] = 0.05 * config["lr_mu_prec"]
    config["lr_w_gamma"] = -1.0

    ## seed everything
    np.random.seed(config["seed"])
    tf.random.set_seed(config["seed"])

    ## Create target dist (GMM): I tested the following target distributions/settings
    # (i) Simple toy target
    config["savepath"] = os.path.join(scriptpath, "log", "toy_gmm")
    config["n_components_model"] = 3
    target_dist = make_simple_target()
    # (ii) Star-GMM from Lin et al. (Khan)
    # config["savepath"] = os.path.join(scriptpath, "log", "star_gmm")
    # config["n_components_model"] = 10
    # target_dist = make_star_target(num_components=5)
    # (iii) Random GMM
    # config["savepath"] = os.path.join(scriptpath, "log", "random_gmm")
    # config["n_components_model"] = 3
    # target_dist = make_target(num_dimensions=2)

    ## create initial GMM parameters
    w_init, mu_init, cov_init = create_initial_model(
        D=config["n_dimensions_model"], K=config["n_components_model"], prior_scale=1.0
    )

    ## define a callback function
    callback = lambda model: plot2d(
        model=model, target_dist=target_dist, fig=fig, axes=axes
    )
    # callback = None  # no callback

    ## fit model
    fig, axes = plt.subplots(nrows=3, ncols=1, squeeze=False, figsize=(3, 8))
    plt.ion()
    model = i_bayesian_learning_rule_gmm(
        config=config,
        target_dist=target_dist,
        w_init=w_init,
        mu_init=mu_init,
        cov_init=cov_init,
        callback=callback,
    )
    # plot final model
    if callback is not None:
        plot2d(model=model, target_dist=target_dist, fig=fig, axes=axes, block=True)


if __name__ == "__main__":
    main()
