import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from i_bayes_rule.i_bayesian_learning_rule_gmm import i_bayesian_learning_rule_gmm
from i_bayes_rule.lnpdf import make_simple_target, make_star_target, make_target


def create_initial_gmm_parameters(
    d_z: int,
    n_tasks: int,
    n_components: int,
    prior_scale: float,
):
    # TODO: check more sophisticated initializations
    prior = tfp.distributions.Normal(
        loc=tf.zeros(d_z), scale=prior_scale * tf.ones(d_z)
    )
    initial_cov = prior_scale**2 * tf.eye(d_z)  # same as prior covariance

    weights = tf.ones((n_tasks, n_components)) / n_components
    means = prior.sample((n_tasks, n_components))
    covs = tf.stack([initial_cov] * n_components, axis=0)
    covs = tf.stack([covs] * n_tasks, axis=0)

    # check output
    assert weights.shape == (n_tasks, n_components)
    assert means.shape == (n_tasks, n_components, d_z)
    assert covs.shape == (n_tasks, n_components, d_z, d_z)
    if n_tasks == 1:  # for backwards compatibility
        weights = tf.squeeze(weights, 0)
        means = tf.squeeze(means, 0)
        covs = tf.squeeze(covs, 0)
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
    xy_tf = tf.convert_to_tensor(xy, dtype=np.float32)

    # determine n_task
    if tf.rank(model.loc) == 2:
        n_tasks = 1
    else:
        assert tf.rank(model.loc) == 3  # not implemented for more than one batch-dim
        n_tasks = model.loc.shape[0]
        xy_tf = tf.broadcast_to(xy_tf[:, None, :], (n_plt**2, n_tasks, 2))

    # evaluate distributions
    p_tgt = np.exp(target_dist.log_density(xy_tf).numpy())
    p_model = np.exp(model.log_density(xy_tf).numpy())

    # extract gmm parameters
    weights = np.exp(model.log_w.numpy())
    locs = model.loc.numpy()
    scale_trils = model.scale_tril.numpy()

    # add task dimension also if n_tasks == 1
    if n_tasks == 1:
        p_tgt = p_tgt[:, None]
        p_model = p_model[:, None]
        weights = weights[None, :]
        scale_trils = scale_trils[None, :, :]
        locs = locs[None, :]

    # plot
    for l in range(n_tasks):
        # plot target distribution
        ax = axes[0, l]
        ax.clear()
        ax.contourf(xx, yy, p_tgt[:, l].reshape(n_plt, n_plt), levels=100)
        ax.axis("scaled")
        ax.set_title("Target density")
        ax.set_xlabel("$z_1$")
        ax.set_ylabel("$z_2$")

        # plot model distribution
        ax = axes[1, l]
        ax.clear()
        ax.contourf(xx, yy, p_model[:, l].reshape(n_plt, n_plt), levels=100)
        colors = []
        for k in range(model.n_components):
            color = next(ax._get_lines.prop_cycler)["color"]
            colors.append(color)
            cur_scale_tril = scale_trils[l, k]
            cur_loc = locs[l, k]
            ax.scatter(x=cur_loc[0], y=cur_loc[1])
            plot_gaussian_ellipse(
                ax=ax, mean=cur_loc, scale_tril=cur_scale_tril, color=color
            )
        ax.axis("scaled")
        ax.set_title("Model density")
        ax.set_xlabel("$z_1$")
        ax.set_ylabel("$z_2$")

        # plot weights
        ax = axes[2, l]
        ax.clear()
        ax.pie(weights[l], labels=[f"{w*100:.2f}%" for w in weights[l]], colors=colors)
        ax.axis("scaled")
        ax.set_title("Mixture weights")

    fig.tight_layout()

    plt.show(block=block)
    plt.pause(0.001)


def main():
    # tf.config.run_functions_eagerly(True)  # only for debugging

    ## create config
    scriptpath = os.path.dirname(os.path.abspath(__file__))
    config = dict()
    config["seed"] = 123
    config["log_interval"] = 500  # how often the parameters are written to npz files
    config["callback_interval"] = 500  # how often the callback is called
    config["prec_update_method"] = "reparam"
    # config["prec_update_method"] = "hessian"
    # The following are the standard parameters from Lin et al. (for the GMM-star exp)
    config["n_iter"] = int(1e4)
    config["n_samples_per_iter"] = 50
    config["lr_mu_prec"] = 0.01
    config["lr_w"] = 0.05 * config["lr_mu_prec"]
    config["use_autograd_for_model"] = False

    ## seed everything
    np.random.seed(config["seed"])
    tf.random.set_seed(config["seed"])

    ## set number of tasks
    n_tasks = 3

    ## Create target dist (GMM): choose one of the following target distributions
    # (i) Simple toy target
    config["savepath"] = os.path.join(scriptpath, "log", "toy_gmm")
    config["n_components_model"] = 3
    config["n_dimensions_model"] = 2
    target_dist = make_simple_target(n_tasks=n_tasks)
    # (ii) Star-GMM from Lin et al.
    # config["savepath"] = os.path.join(scriptpath, "log", "star_gmm")
    # config["n_components_model"] = 10
    # config["n_dimensions_model"] = 2
    # target_dist = make_star_target(num_components=5)
    # (iii) Random GMM
    # config["savepath"] = os.path.join(scriptpath, "log", "random_gmm")
    # n_dimensions = 200
    # config["n_components_model"] = 3
    # config["n_dimensions_model"] = n_dimensions
    # target_dist = make_target(num_dimensions=n_dimensions)

    ## create initial GMM parameters
    w_init, mu_init, cov_init = create_initial_gmm_parameters(
        n_tasks=n_tasks,
        d_z=config["n_dimensions_model"],
        n_components=config["n_components_model"],
        prior_scale=1.0,
    )

    ## define a callback function
    if config["n_dimensions_model"] == 2:
        callback = lambda model: plot2d(
            model=model, target_dist=target_dist, fig=fig, axes=axes
        )
    else:
        callback = None  # no callback

    ## fit model
    fig, axes = plt.subplots(
        nrows=3, ncols=n_tasks, squeeze=False, figsize=(3 * n_tasks, 8)
    )
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
