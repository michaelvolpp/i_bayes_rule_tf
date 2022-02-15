import numpy as np
from metalearning_benchmarks.benchmarks.affine1d_benchmark import Affine1D
from metalearning_benchmarks.benchmarks.sinusoid_benchmark import Sinusoid
from metalearning_benchmarks.benchmarks.quadratic1d_benchmark import Quadratic1D
from metalearning_benchmarks.benchmarks.base_benchmark import MetaLearningBenchmark
import tensorflow as tf
from aggbyopt_nps.np import meta_train_np, test_np, NP
from matplotlib import pyplot as plt
from tqdm import tqdm
from aggbyopt_nps.np import PosteriorLearner


def create_tf_dataset(benchmark: MetaLearningBenchmark):
    x = np.zeros(
        (benchmark.n_task, benchmark.n_datapoints_per_task, benchmark.d_x),
        dtype=np.float32,
    )
    y = np.zeros(
        (benchmark.n_task, benchmark.n_datapoints_per_task, benchmark.d_x),
        dtype=np.float32,
    )
    for l, task in enumerate(benchmark):
        x[l] = task.x
        y[l] = task.y
    dataset = tf.data.Dataset.from_tensor_slices((x, y))

    return dataset


def plot2d(
    np_model: NP, x: tf.Tensor, y: tf.Tensor, n_task_max: int, fig, axes, block=False
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

    # create meshgrid
    n_plt = 25
    loc_min = tf.reduce_min(np_model.gmm.loc)
    loc_max = tf.reduce_max(np_model.gmm.loc)
    z_min = loc_min - 1.0 * (loc_max - loc_min)
    z_max = loc_max + 1.0 * (loc_max - loc_min)
    z1 = np.linspace(z_min, z_max, n_plt)
    z2 = np.linspace(z_min, z_max, n_plt)
    zz1, zz2 = np.meshgrid(z1, z2)
    z_grid = np.vstack([zz1.ravel(), zz2.ravel()]).T
    z_grid = tf.convert_to_tensor(z_grid, dtype=np.float32)
    z_grid = tf.broadcast_to(z_grid[:, None, :], (n_plt ** 2, np_model.n_task, 2))

    # evaluate distributions
    # TODO: currently, we always have to make predictions for all tasks in the model
    p_tgt = np.exp(
        np_model.log_unnormalized_posterior_density(x=x, y=y, z=z_grid).numpy()
    )
    p_gmm = np.exp(np_model.log_approximate_posterior_density(z=z_grid).numpy())
    gmm_weights = np.exp(np_model.gmm.log_w.numpy())

    # evaluate predictions
    n_samples = 10
    z_samples = np_model.sample_approximate_posterior(n_samples)
    x_min = tf.reduce_min(x)
    x_max = tf.reduce_max(x)
    x_plt_min = x_min - 0.25 * (x_max - x_min)
    x_plt_max = x_max + 0.25 * (x_max - x_min)
    x_plt = tf.linspace(x_plt_min, x_plt_max, 128)
    x_plt = tf.reshape(x_plt, (-1, 1))
    x_plt = tf.repeat(x_plt[None, :, :], repeats=np_model.n_task, axis=0)
    mu = np_model.predict(x=x_plt, z=z_samples)

    for l in range(n_task_plot):
        # plot target distribution
        ax = axes[0, l]
        ax.clear()
        ax.contourf(zz1, zz2, p_tgt[:, l].reshape(n_plt, n_plt), levels=100)
        ax.axis("scaled")
        ax.set_title("Target density")
        ax.set_xlabel("$z_1$")
        ax.set_ylabel("$z_2$")

        # plot model distribution
        ax = axes[1, l]
        ax.clear()
        ax.contourf(zz1, zz2, p_gmm[:, l].reshape(n_plt, n_plt), levels=100)
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
            ax.plot(x_plt[l], mu[s, l], color="b", alpha=0.3)
        ax.grid()
        ax.set_title("Predictions")

    fig.tight_layout()

    plt.show(block=block)
    plt.pause(0.001)


def main():
    # config
    config = dict()
    config["seed"] = 1234
    config["data_noise_std"] = 1.0
    config["n_task_meta"] = 16
    config["n_datapoints_per_task_meta"] = 16
    config["seed_task_meta"] = 1234
    config["seed_x_meta"] = 2234
    config["seed_noise_meta"] = 3234
    config["n_task_test"] = 4
    config["n_datapoints_per_task_test"] = 2
    config["seed_task_test"] = 1235
    config["seed_x_test"] = 2235
    config["seed_noise_test"] = 3235
    config["d_z"] = 2
    config["gmm_n_components"] = 2
    config["gmm_prior_scale"] = 1.0
    config["gmm_learner_lr_mu_prec"] = 0.01
    config["gmm_learner_lr_w"] = 0.05 * config["gmm_learner_lr_mu_prec"]
    config["gmm_learner_n_samples"] = 10
    config["model_prior_scale"] = 1.0
    config["decoder_n_hidden"] = 2
    config["decoder_d_hidden"] = 10
    # TODO: how to choose output noise
    config["decoder_output_scale"] = config["data_noise_std"] * 1.0
    config["decoder_learner_lr"] = 0.01
    config["decoder_learner_n_samples"] = 10
    config["n_iter"] = 1
    config["n_steps_per_iter"] = 1000
    # TODO: how to adjust this?
    config["n_theta_steps_per_gmm_step"] = 1

    # load meta benchmark and generate dataset
    benchmark_meta = Affine1D(
        n_task=config["n_task_meta"],
        n_datapoints_per_task=config["n_datapoints_per_task_meta"],
        output_noise=config["data_noise_std"],
        seed_task=config["seed_task_meta"],
        seed_x=config["seed_x_meta"],
        seed_noise=config["seed_noise_meta"],
    )
    dataset_meta = create_tf_dataset(benchmark_meta)
    config["d_x"] = benchmark_meta.d_x
    config["d_y"] = benchmark_meta.d_y

    # load test benchmark and generate dataset
    benchmark_test = Affine1D(
        n_task=config["n_task_test"],
        n_datapoints_per_task=config["n_datapoints_per_task_test"],
        output_noise=config["data_noise_std"],
        seed_task=config["seed_task_test"],
        seed_x=config["seed_x_test"],
        seed_noise=config["seed_noise_test"],
    )
    dataset_test = create_tf_dataset(benchmark_test)

    # seed
    np.random.seed(config["seed"])
    tf.random.set_seed(config["seed"])

    ## train
    # tf.config.run_functions_eagerly(True)  # only for debugging
    n_task_max = 4
    fig, axes = plt.subplots(nrows=4, ncols=n_task_max, squeeze=False, figsize=(15, 8))
    callback = lambda iteration, np_model, x, y: plot2d(
        np_model=np_model, n_task_max=n_task_max, x=x, y=y, fig=fig, axes=axes
    )
    np_model = meta_train_np(
        config=config, dataset_meta=dataset_meta, callback=callback
    )

    ## test
    cp_learner = PosteriorLearner(
        model=np_model,
        lr_w=config["gmm_learner_lr_w"],
        lr_mu_prec=config["gmm_learner_lr_mu_prec"],
        n_samples=config["gmm_learner_n_samples"],
    )
    n_task = config["n_task_test"]

    # load data
    dataset_test = dataset_test.batch(n_task)
    for batch in dataset_test:
        x_test, y_test = batch

    # adapt model
    np_model.reset_gmm(n_tasks=n_task)
    for _ in tqdm(range(config["n_steps_per_iter"])):
        cp_learner.step(x=x_test, y=y_test)

    # make predictions
    fig, axes = plt.subplots(nrows=4, ncols=n_task_max, squeeze=False, figsize=(15, 8))
    plot2d(
        np_model=np_model,
        n_task_max=config["n_task_test"],
        x=x_test,
        y=y_test,
        fig=fig,
        axes=axes,
    )
    plt.show()


if __name__ == "__main__":
    main()
