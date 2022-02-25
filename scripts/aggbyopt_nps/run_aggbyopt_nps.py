import os

import numpy as np
import tensorflow as tf
import wandb
from matplotlib import pyplot as plt
from metalearning_benchmarks.benchmarks.affine1d_benchmark import Affine1D
from metalearning_benchmarks.benchmarks.base_benchmark import MetaLearningBenchmark
from metalearning_benchmarks.benchmarks.quadratic1d_benchmark import Quadratic1D
from metalearning_benchmarks.benchmarks.sinusoid_benchmark import Sinusoid
from metalearning_eval_util.util import compute_log_marginal_likelihood_mc, compute_mse

from aggbyopt_nps.np import NP, adapt_np, meta_train_np

BM_DICT = {"Affine1D": Affine1D, "Quadratic1D": Quadratic1D, "Sinusoid1D": Sinusoid}


def collate_benchmark(benchmark: MetaLearningBenchmark):
    # collate test data
    x = np.zeros((benchmark.n_task, benchmark.n_datapoints_per_task, benchmark.d_x))
    y = np.zeros((benchmark.n_task, benchmark.n_datapoints_per_task, benchmark.d_y))
    for l, task in enumerate(benchmark):
        x[l] = task.x
        y[l] = task.y

    return x, y


def callback_fn(
    iteration: int,
    np_model: NP,
    metrics: dict,
    benchmark: MetaLearningBenchmark,
    config: dict,
    wandb_tag: str,
    wandb_run,
):
    """ Log stuff to wandb. """
    log_dict = {"iter": iteration}
    if metrics is not None:
        log_dict.update({f"{wandb_tag}_{k}": v for k, v in metrics.items()})
    if iteration % config["plot_interval"] == 0 or iteration == config["n_iter"] - 1:
        fig = plot(
            np_model=np_model,
            n_task_max=config["n_task_plot"],
            benchmark=benchmark,
        )
        log_dict.update({f"{wandb_tag}_plot": wandb.Image(fig)})
        if wandb_run.mode !="disabled":
            plt.close(fig)
    wandb_run.log(log_dict)


def plot(
    np_model: NP,
    benchmark: MetaLearningBenchmark,
    n_task_max: int,
):
    def plot_gaussian_ellipse(ax, mean, scale_tril, color):
        n_plot = 100
        evals, evecs = np.linalg.eig(scale_tril @ scale_tril.T)
        theta = np.linspace(0, 2 * np.pi, n_plot)
        ellipsis = (np.sqrt(evals[None, :]) * evecs) @ [np.sin(theta), np.cos(theta)]
        ellipsis = ellipsis + mean[:, None]
        ax.plot(ellipsis[0, :], ellipsis[1, :], color=color)

    # determine n_task
    n_task_plot = min(n_task_max, np_model.n_task)

    # prepare plot
    fig, axes = plt.subplots(nrows=4, ncols=n_task_plot, squeeze=False, figsize=(15, 8))

    # create x, y, z
    x, y = collate_benchmark(benchmark=benchmark)
    n_ticks = 25
    loc_min = tf.reduce_min(np_model.gmm.loc)
    loc_max = tf.reduce_max(np_model.gmm.loc)
    z_min = loc_min - 1.0 * (loc_max - loc_min)
    z_max = loc_max + 1.0 * (loc_max - loc_min)
    z1 = np.linspace(z_min, z_max, n_ticks)
    z2 = np.linspace(z_min, z_max, n_ticks)
    zz1, zz2 = np.meshgrid(z1, z2)
    z_grid = np.vstack([zz1.ravel(), zz2.ravel()]).T
    z_grid = tf.convert_to_tensor(z_grid, dtype=np.float32)
    z_grid = tf.broadcast_to(z_grid[:, None, :], (n_ticks ** 2, np_model.n_task, 2))

    # evaluate distributions on z-grid
    # TODO: currently, we always have to make predictions for all tasks in the model
    p_tgt = np.exp(
        np_model._log_unnormalized_posterior_density(
            x=tf.constant(x, dtype=tf.float32),
            y=tf.constant(y, dtype=tf.float32),
            z=z_grid,
        ).numpy()
    )
    p_gmm = np.exp(np_model._log_approximate_posterior_density(z=z_grid).numpy())
    gmm_weights = np.exp(np_model.gmm.log_w.numpy())

    # evaluate predictions
    n_samples = 10
    x_min = np.min(x)
    x_max = np.max(x)
    x_plt_min = x_min - 0.25 * (x_max - x_min)
    x_plt_max = x_max + 0.25 * (x_max - x_min)
    x_plt = np.linspace(x_plt_min, x_plt_max, 128)
    x_plt = np.reshape(x_plt, (-1, 1))
    x_plt = np.repeat(x_plt[None, :, :], repeats=np_model.n_task, axis=0)
    mu_post, _ = np_model.predict(
        x=x_plt, n_samples=n_samples, sample_from="approximate_posterior"
    )
    mu_prior, _ = np_model.predict(x=x_plt, n_samples=n_samples, sample_from="prior")

    for l in range(n_task_plot):
        # plot target distribution
        ax = axes[0, l]
        ax.clear()
        ax.contourf(zz1, zz2, p_tgt[:, l].reshape(n_ticks, n_ticks), levels=100)
        ax.axis("scaled")
        ax.set_title("Target density")
        ax.set_xlabel("$z_1$")
        ax.set_ylabel("$z_2$")

        # plot model distribution
        ax = axes[1, l]
        ax.clear()
        ax.contourf(zz1, zz2, p_gmm[:, l].reshape(n_ticks, n_ticks), levels=100)
        colors = []
        for k in range(np_model.gmm.n_components):
            color = next(ax._get_lines.prop_cycler)["color"]
            colors.append(color)
            scale_tril = np_model.gmm.scale_tril[l, k].numpy()
            mean = np_model.gmm.loc[l, k].numpy()
            ax.scatter(x=mean[0], y=mean[1])
            plot_gaussian_ellipse(ax=ax, mean=mean, scale_tril=scale_tril, color=color)
        ax.axis("scaled")
        ax.set_title("Model density")
        ax.set_xlabel("$z_1$")
        ax.set_ylabel("$z_2$")

        # plot weights
        ax = axes[2, l]
        ax.clear()
        ax.pie(
            gmm_weights[l],
            labels=[f"{w*100:.2f}%" for w in gmm_weights[l]],
            colors=colors,
        )
        ax.axis("scaled")
        ax.set_title("Mixture weights")

        # plot predictions
        ax = axes[3, l]
        ax.sharex(axes[3, 0])
        ax.sharey(axes[3, 0])
        ax.clear()
        ax.scatter(x[l], y[l], marker="x", s=5, color="r")
        for s in range(n_samples):
            ax.plot(x_plt[l], mu_post[s, l], color="b", alpha=0.3, label="posterior")
            ax.plot(x_plt[l], mu_prior[s, l], color="g", alpha=0.3, label="prior")
        ax.grid()
        ax.set_title("Predictions")

    fig.tight_layout()
    return fig


def run_experiment(config, wandb_run):
    ## generate benchmarks
    benchmark_meta = BM_DICT[config["benchmark"]](
        n_task=config["n_task_meta"],
        n_datapoints_per_task=config["n_datapoints_per_task_meta"],
        output_noise=config["data_noise_std"],
        seed_task=config["seed_task_meta"],
        seed_x=config["seed_x_meta"],
        seed_noise=config["seed_noise_meta"],
    )
    benchmark_test = BM_DICT[config["benchmark"]](
        n_task=config["n_task_test"],
        n_datapoints_per_task=config["n_datapoints_per_task_test"],
        output_noise=config["data_noise_std"],
        seed_task=config["seed_task_test"],
        seed_x=config["seed_x_test"],
        seed_noise=config["seed_noise_test"],
    )

    ## seed
    np.random.seed(config["seed"])
    tf.random.set_seed(config["seed"])

    ## create model
    np_model = NP(
        d_x=benchmark_meta.d_x,
        d_y=benchmark_meta.d_y,
        d_z=config["d_z"],
        gmm_n_components=config["gmm_n_components"],
        gmm_prior_scale=config["gmm_prior_scale"],
        latent_prior_scale=config["latent_prior_scale"],
        decoder_n_hidden=config["decoder_n_hidden"],
        decoder_d_hidden=config["decoder_d_hidden"],
        decoder_output_scale=config["decoder_output_scale"],
    )

    ## train
    callback = lambda iteration, np_model, metrics: callback_fn(
        iteration=iteration,
        np_model=np_model,
        metrics=metrics,
        benchmark=benchmark_meta,
        config=config,
        wandb_tag="train",
        wandb_run=wandb_run,
    )
    np_model = meta_train_np(
        np_model=np_model,
        benchmark=benchmark_meta,
        n_iter=config["n_iter"],
        n_tasks=benchmark_meta.n_task,
        gmm_learner_lr_w=config["gmm_learner_lr_w"],
        gmm_learner_lr_mu_prec=config["gmm_learner_lr_mu_prec"],
        gmm_learner_n_samples=config["gmm_learner_n_samples"],
        decoder_learner_lr=config["decoder_learner_lr"],
        decoder_learner_n_samples=config["decoder_learner_n_samples"],
        decoder_learner_n_context=config["decoder_learner_n_context"],
        callback=callback,
    )

    ## test
    callback = lambda iteration, np_model, metrics: callback_fn(
        iteration=iteration,
        np_model=np_model,
        metrics=metrics,
        benchmark=benchmark_test,
        config=config,
        wandb_tag="test",
        wandb_run=wandb_run,
    )
    np_model = adapt_np(
        np_model=np_model,
        benchmark=benchmark_test,
        n_iter=config["n_iter"],
        n_tasks=benchmark_test.n_task,
        gmm_learner_lr_w=config["gmm_learner_lr_w"],
        gmm_learner_lr_mu_prec=config["gmm_learner_lr_mu_prec"],
        gmm_learner_n_samples=config["gmm_learner_n_samples"],
        callback=callback,
    )

    ## compute metrics
    # collate data
    x_test, y_test = collate_benchmark(benchmark=benchmark_test)
    # predict
    y_pred, var_pred = np_model.predict(
        x=x_test,
        n_samples=config["n_samples_test"],
        sample_from="approximate_posterior",
    )
    # compute metrics
    mse = compute_mse(y_pred=y_pred, y_true=y_test)
    lmlhd = compute_log_marginal_likelihood_mc(
        y_pred=y_pred, sigma_pred=np.sqrt(var_pred), y_true=y_test
    )
    # log metrics
    wandb_run.summary["test_mse"] = mse
    wandb_run.summary["test_lmlhd"] = lmlhd
    print(f"MSE   = {mse:.4f}")
    print(f"LMLHD = {lmlhd:.4f}")

    # show all plots
    if wandb_run.mode == "disabled":
        plt.show()


def main():
    # tf.config.run_functions_eagerly(True)  # only for debugging
    ## wandb
    wandb_mode = os.getenv("WANDB_MODE", "online")
    smoke_test = os.getenv("SMOKE_TEST", "False") == "True"
    print(f"wandb_mode={wandb_mode}")
    print(f"smoke_test={smoke_test}")

    ## config
    config = dict()
    # model name
    config["model"] = "AggByOptNP"
    config["smoke_test"] = smoke_test
    # seed
    config["seed"] = 1234
    # benchmark
    config["benchmark"] = "Quadratic1D"
    config["data_noise_std"] = 0.5
    # meta data
    config["n_task_meta"] = 64
    config["n_datapoints_per_task_meta"] = 16
    config["seed_task_meta"] = 1234
    config["seed_x_meta"] = 2234
    config["seed_noise_meta"] = 3234
    # test data
    config["n_task_test"] = 64
    config["n_datapoints_per_task_test"] = 16
    config["seed_task_test"] = 1235
    config["seed_x_test"] = 2235
    config["seed_noise_test"] = 3235
    # architecture
    config["d_z"] = 2
    config["gmm_n_components"] = 2
    config["gmm_prior_scale"] = 1.0  # to initialize the GMM
    config["latent_prior_scale"] = 1.0
    config["decoder_n_hidden"] = 2
    config["decoder_d_hidden"] = 16
    config["decoder_output_scale"] = config["data_noise_std"]
    # training
    config["n_iter"] = 100 if smoke_test else 5000
    config["gmm_learner_lr_mu_prec"] = 0.01
    config["gmm_learner_lr_w"] = 0.05 * config["gmm_learner_lr_mu_prec"]
    config["gmm_learner_n_samples"] = 16
    config["decoder_learner_lr"] = 0.01
    config["decoder_learner_n_samples"] = 16
    # config["decoder_learner_n_context"] = (
    #     config["n_datapoints_per_task_test"],
    #     config["n_datapoints_per_task_test"],
    # )
    config["decoder_learner_n_context"] = (0, 0)
    # testing
    config["n_samples_test"] = 1024
    # plotting/logging
    config["plot_interval"] = config["n_iter"] // 1 if smoke_test else config["n_iter"] // 5
    config["n_task_plot"] = 5

    ## run
    if wandb_mode != "disabled":
        wandb.login()
    with wandb.init(
        project="aggbyopt_nps", mode=wandb_mode, config=config
    ) as wandb_run:
        config = wandb_run.config
        run_experiment(config=config, wandb_run=wandb_run)


if __name__ == "__main__":
    main()
