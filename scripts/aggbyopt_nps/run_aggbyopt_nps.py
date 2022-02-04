import numpy as np
from metalearning_benchmarks.benchmarks.quadratic1d_benchmark import Quadratic1D
from metalearning_benchmarks.benchmarks.base_benchmark import MetaLearningBenchmark
import tensorflow as tf
from aggbyopt_nps.np import meta_train_np, test_np


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
config["data_noise_std"] = 0.1
config["n_task_meta"] = 8
config["n_datapoints_per_task_meta"] = 16
config["seed_task_meta"] = 1234
config["seed_x_meta"] = 2234
config["seed_noise_meta"] = 3234
config["n_task_test"] = 4
config["n_datapoints_per_task_test"] = 128
config["seed_task_test"] = 1235
config["seed_x_test"] = 2235
config["seed_noise_test"] = 3235
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
config["batch_size"] = config["n_task_meta"]
config["n_context"] = 8
config["n_iter"] = 1
config["steps_per_iter"] = 10000

# load meta benchmark and generate dataset
benchmark_meta = Quadratic1D(
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
benchmark_test = Quadratic1D(
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

# train
# tf.config.run_functions_eagerly(True)  # only for debugging
np_model = meta_train_np(config=config, dataset_meta=dataset_meta, callback=None)

# test
test_np(config=config, n_context=8, np_model=np_model, dataset_test=dataset_test)
