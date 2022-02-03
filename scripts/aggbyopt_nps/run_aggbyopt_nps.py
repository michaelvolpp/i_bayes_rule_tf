import numpy as np
from metalearning_benchmarks.benchmarks.quadratic1d_benchmark import Quadratic1D
from metalearning_benchmarks.benchmarks.base_benchmark import MetaLearningBenchmark
import tensorflow as tf
from aggbyopt_nps.np import meta_train_np


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


# config
config = dict()
config["seed"] = 1234
config["n_task_meta"] = 8
config["n_datapoints_per_task"] = 16
config["data_noise_std"] = 0.1
config["seed_task_meta"] = config["seed"] + 1000
config["seed_x_meta"] = config["seed"] + 2000
config["seed_noise_meta"] = config["seed"] + 3000
config["d_z"] = 3
config["gmm_n_components"] = 1
config["gmm_prior_scale"] = 1.0
config["gmm_learner_lr_mu_prec"] = 0.01
config["gmm_learner_lr_w"] = 0.05 * config["gmm_learner_lr_mu_prec"]
config["gmm_learner_n_samples"] = 10
config["decoder_n_hidden"] = 0
config["decoder_d_hidden"] = None
config["decoder_output_scale"] = config["data_noise_std"]
config["decoder_learner_lr"] = 0.01
config["decoder_learner_n_samples"] = 10
config["batch_size"] = 4
config["n_context"] = 8
config["n_iter"] = 100
config["steps_per_batch"] = 100

# load benchmark and generate dataloader
n_task = config["n_task_meta"]
n_datapoints_per_task = config["n_datapoints_per_task"]
benchmark = Quadratic1D(
    n_task=n_task,
    n_datapoints_per_task=n_datapoints_per_task,
    output_noise=config["data_noise_std"],
    seed_task=config["seed_task_meta"],
    seed_x=config["seed_x_meta"],
    seed_noise=config["seed_noise_meta"],
)
meta_dataset = create_tf_dataset(benchmark)
config["d_x"] = benchmark.d_x
config["d_y"] = benchmark.d_y

# seed
np.random.seed(config["seed"])
tf.random.set_seed(config["seed"])

# train
# tf.config.run_functions_eagerly(True)  # only for debugging
np_model = meta_train_np(config=config, meta_dataset=meta_dataset, callback=None)
