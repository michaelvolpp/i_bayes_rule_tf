from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from i_bayes_rule.i_bayesian_learning_rule_gmm import step as gmm_learner_step
from i_bayes_rule.util import GMM
from metalearning_benchmarks.benchmarks.base_benchmark import MetaLearningBenchmark
from tensorflow import keras
from tqdm import tqdm


# TODO: check that all functions decorated with tf.function always are called with the same tensor-shapes!
# TODO: normalize data (and z)
class NP:
    """
    p(D^t | theta) = \int p(D^t | z, theta) p(z | theta) dz
    """

    def __init__(
        self,
        d_x: int,
        d_y: int,
        d_z: int,
        gmm_n_components: int,
        gmm_prior_scale: float,
        latent_prior_scale: float,
        decoder_n_hidden: int,
        decoder_d_hidden: int,
        decoder_output_scale: float,
    ):
        self.d_x = d_x
        self.d_y = d_y
        self.d_z = d_z
        self.gmm_n_components = gmm_n_components
        self.gmm_prior_scale = gmm_prior_scale
        self.latent_prior_scale = latent_prior_scale
        self.decoder_n_hidden = decoder_n_hidden
        self.decoder_d_hidden = decoder_d_hidden
        self.decoder_output_scale = decoder_output_scale

        self.decoder = create_mlp(
            d_x=self.d_x + self.d_z,
            d_y=self.d_y,
            n_hidden=self.decoder_n_hidden,
            d_hidden=self.decoder_d_hidden,
        )
        self.n_task = None
        self.gmm = None

        # TODO: improve this -> only required for gradient/Hessian computation
        self.x, self.y = None, None

    @tf.function
    def _predict(self, x: tf.Tensor, z: tf.Tensor):
        # check input
        n_points = x.shape[1]
        n_samples = z.shape[0]
        assert x.shape == (self.n_task, n_points, self.d_x)
        assert z.shape == (n_samples, self.n_task, self.d_z)

        # TODO: do we need the broadcasting or can this be done automatically by keras?
        x = tf.broadcast_to(x[None, ...], (n_samples, self.n_task, n_points, self.d_x))
        z = tf.broadcast_to(
            z[..., None, :], (n_samples, self.n_task, n_points, self.d_z)
        )
        xz = tf.concat((x, z), axis=-1)
        assert xz.shape == (n_samples, self.n_task, n_points, self.d_x + self.d_z)
        mu = self.decoder(xz)

        # compute output variance
        var = self.decoder_output_scale ** 2
        var = tf.broadcast_to(var, mu.shape)

        # check output
        assert mu.shape == (n_samples, self.n_task, n_points, self.d_y)
        return mu, var

    @tf.function
    def _log_likelihood(self, x: tf.Tensor, y: tf.Tensor, z: tf.Tensor):
        """
        log p(D^t | z, theta)
        """
        # check input
        n_points = x.shape[1]
        n_samples = z.shape[0]
        assert x.shape == (self.n_task, n_points, self.d_x)
        assert y.shape == (self.n_task, n_points, self.d_y)
        assert z.shape == (n_samples, self.n_task, self.d_z)

        # compute log likelihood
        y = tf.broadcast_to(y[None, ...], (n_samples, self.n_task, n_points, self.d_y))
        mu, var = self._predict(x=x, z=z)
        gaussian = tfp.distributions.Independent(
            tfp.distributions.Normal(loc=mu, scale=tf.sqrt(var)),
            reinterpreted_batch_ndims=1,  # sum ll of data dim upon calling log_prob
        )
        log_likelihood = gaussian.log_prob(y)
        log_likelihood = tf.reduce_sum(
            log_likelihood, axis=-1
        )  # TODO: is this correct? sum ll of datapoints

        # check output
        assert log_likelihood.shape == (n_samples, self.n_task)
        return log_likelihood

    @tf.function
    def _log_prior_density(self, z: tf.Tensor):
        """
        log p(z | theta)
        """
        # TODO: learn this prior?

        # check input
        n_samples = z.shape[0]
        assert z.shape == (n_samples, self.n_task, self.d_z)

        # compute log conditional prior density
        gaussian = tfp.distributions.Independent(
            tfp.distributions.Normal(
                loc=tf.zeros((self.d_z,)),
                scale=self.latent_prior_scale * tf.ones((self.d_z,)),
            ),
            reinterpreted_batch_ndims=1,  # sum ll of z-dim upon calling log_prob
        )
        log_prior_density = gaussian.log_prob(z)

        # check output
        assert log_prior_density.shape == (n_samples, self.n_task)
        return log_prior_density

    @tf.function
    def _log_unnormalized_posterior_density(
        self, x: tf.Tensor, y: tf.Tensor, z: tf.Tensor
    ):
        """
        log p(D^t | z, theta) + log p(z | theta)
        """
        # check input
        n_points = x.shape[1]
        n_samples = z.shape[0]
        assert x.shape == (self.n_task, n_points, self.d_x)
        assert y.shape == (self.n_task, n_points, self.d_y)
        assert z.shape == (n_samples, self.n_task, self.d_z)

        # compute log_density
        log_likelihood = self._log_likelihood(x=x, y=y, z=z)
        assert log_likelihood.shape == (n_samples, self.n_task)
        log_prior_density = self._log_prior_density(z=z)
        assert log_prior_density.shape == (n_samples, self.n_task)
        log_density = log_likelihood + log_prior_density

        # check output
        assert log_density.shape == (n_samples, self.n_task)
        return log_density

    @tf.function
    def _log_unnormalized_posterior_density_self_data(self, z: tf.Tensor):
        # This function can be passed to gradient/hessian computations without
        # building a lambda function.
        return self._log_unnormalized_posterior_density(x=self.x, y=self.y, z=z)

    @tf.function
    def _log_approximate_posterior_density(self, z: tf.Tensor):
        # check input
        n_samples = z.shape[0]
        n_task = z.shape[1]
        d_z = z.shape[2]

        log_density = self.gmm.log_density(z=z)

        # check output
        assert log_density.shape == (n_samples, n_task)
        return log_density

    # @tf.function
    # TODO: decorating this with tf.function does not allow to change n_tasks -> why?
    def _sample_approximate_posterior(self, n_samples: int):
        z = self.gmm.sample(n_samples=n_samples)

        # check output
        assert z.shape == (n_samples, self.n_task, self.d_z)
        return z

    # @tf.function
    # TODO: decorating this with tf.function does not allow to change n_tasks -> why?
    def _sample_prior(self, n_samples: int):
        gaussian = tfp.distributions.Independent(
            tfp.distributions.Normal(
                loc=tf.zeros(
                    (
                        self.n_task,
                        self.d_z,
                    )
                ),
                scale=self.latent_prior_scale
                * tf.ones(
                    (
                        self.n_task,
                        self.d_z,
                    )
                ),
            ),
            reinterpreted_batch_ndims=1,  # sum ll of z-dim upon calling log_prob
        )
        z = gaussian.sample(n_samples)

        # check output
        assert z.shape == (n_samples, self.n_task, self.d_z)
        return z

    def reset_gmm(self, n_tasks: int):
        # TODO: in order to learn a prior, this has to be deterministic and the
        #  same for each task!
        self.n_task = n_tasks
        gmm_weights, gmm_means, gmm_covs = create_initial_gmm_parameters(
            n_tasks=n_tasks,
            d_z=self.d_z,
            n_components=self.gmm_n_components,
            prior_scale=self.gmm_prior_scale,
        )
        # TODO: do not compute the inverse here!
        self.gmm = GMM(
            log_w=tf.math.log(gmm_weights),
            loc=gmm_means,
            prec=tf.linalg.inv(gmm_covs),
        )

    def predict(
        self, x: np.ndarray, n_samples: int, sample_from="approximate_posterior"
    ):
        # check input
        n_points_per_task = x.shape[1]
        assert x.shape == (self.n_task, n_points_per_task, self.d_x)

        # convert input to tf
        x = tf.constant(x, dtype=tf.float32)

        # sample z
        if sample_from == "approximate_posterior":
            z = self._sample_approximate_posterior(n_samples=n_samples)
        elif sample_from == "prior":
            z = self._sample_prior(n_samples=n_samples)
        else:
            raise ValueError(f"Unknown value of argument 'sample_from' = {sample_from}")

        # perform prediction
        y_pred, var_y = self._predict(x=x, z=z)

        # convert output back to numpy
        y_pred, var_y = y_pred.numpy(), var_y

        # check output
        assert y_pred.shape == (n_samples, self.n_task, n_points_per_task, self.d_y)
        assert var_y.shape == (n_samples, self.n_task, n_points_per_task, self.d_y)
        return y_pred, var_y


class PosteriorLearner:
    def __init__(self, model: NP, lr_w: float, lr_mu_prec: float, n_samples: int):
        self.model = model
        self.lr_w = lr_w
        self.lr_mu_prec = lr_mu_prec
        self.n_samples = n_samples

    def step(self, x: tf.Tensor, y: tf.Tensor):
        ## check input
        n_points_tgt = x.shape[1]
        assert x.shape == (self.model.n_task, n_points_tgt, self.model.d_x)
        assert y.shape == (self.model.n_task, n_points_tgt, self.model.d_y)

        ## step
        # TODO: is there a better way than setting model attributes here?
        self.model.x = x
        self.model.y = y
        model, _ = gmm_learner_step(
            model=self.model.gmm,
            target_density_fn=self.model._log_unnormalized_posterior_density_self_data,
            n_samples=self.n_samples,
            lr_w=self.lr_w,
            lr_mu_prec=self.lr_mu_prec,
            prec_method="hessian",
            use_autograd_for_model=False,
        )


class DecoderLearner:
    def __init__(
        self,
        model: NP,
        lr: float,
        n_samples: int,
        n_context_min: int,
        n_context_max: int,
    ):
        self.model = model
        self.lr = lr
        self.n_samples = n_samples
        self.n_ctx_min = n_context_min
        self.n_ctx_max = n_context_max
        self.optim = keras.optimizers.Adam(learning_rate=self.lr)

    def step(self, x: tf.Tensor, y: tf.Tensor):
        ## check input
        n_points = x.shape[1]
        assert x.shape == (self.model.n_task, n_points, self.model.d_x)
        assert y.shape == (self.model.n_task, n_points, self.model.d_y)

        # generate context set
        # TODO: understand why we have to do this outside of tf.function
        #  inside tf.function, indexing a tensor yields a None dimension
        #  e.g.: x[:, :2, :].shape = (..., None, ...)
        n_ctx = tf.random.uniform(
            shape=(1,),
            minval=self.n_ctx_min,
            maxval=self.n_ctx_max + 1,  # include self.n_ctx_max
            dtype=tf.int32,
        )[0]
        ctx_idx = tf.random.shuffle(tf.range(0, n_points, dtype=tf.int32))[:n_ctx]
        x_ctx = tf.gather(x, indices=ctx_idx, axis=1)
        y_ctx = tf.gather(y, indices=ctx_idx, axis=1)
        assert x_ctx.shape == (self.model.n_task, n_ctx, self.model.d_x)
        assert y_ctx.shape == (self.model.n_task, n_ctx, self.model.d_y)

        # compute loss
        loss, log_model_density, log_marg_lhd = self.step_tf(
            x=x, y=y, x_ctx=x_ctx, y_ctx=y_ctx
        )

        # return some metrics
        metrics = {
            "loss_theta": loss.numpy(),
            "log_model_density": log_model_density.numpy(),
            "log_marginal_likelihood": log_marg_lhd.numpy(),
        }
        return metrics

    @tf.function
    def step_tf(self, x: tf.Tensor, y: tf.Tensor, x_ctx: tf.Tensor, y_ctx: tf.Tensor):
        ## perform step
        # sample model
        z_post = self.model._sample_approximate_posterior(n_samples=self.n_samples)
        # TODO: use importance sampling to re-use z_post?
        # TODO: if the prior is a fct of theta, these gradients won't be recorded here
        z_prior = self.model._sample_prior(n_samples=self.n_samples)
        assert z_post.shape == (self.n_samples, self.model.n_task, self.model.d_z)
        assert z_prior.shape == (self.n_samples, self.model.n_task, self.model.d_z)
        with tf.GradientTape() as tape:
            # compute expectation of unnormalized posterior under approx. posterior
            #  we drop the entropy of the approx. posterior as it is constant in theta
            log_model_density = tf.reduce_mean(
                self.model._log_unnormalized_posterior_density(x=x, y=y, z=z_post)
            )
            if (
                x_ctx[1].shape != 0
            ):  # TODO: check that this is handled correctly in graph mode
                # compute log marginal likelihood of context set
                #  we drop the log 1/S term as it is constant in theta
                log_marg_lhd = self.model._log_unnormalized_posterior_density(
                    x=x_ctx,
                    y=y_ctx,
                    z=z_prior,
                )
                assert log_marg_lhd.shape == (self.n_samples, self.model.n_task)
                log_marg_lhd = tf.reduce_logsumexp(log_marg_lhd, axis=0, keepdims=True)
                log_marg_lhd = tf.reduce_mean(log_marg_lhd, axis=1, keepdims=True)
                assert log_marg_lhd.shape == (1, 1)
                log_marg_lhd = tf.squeeze(log_marg_lhd)
            else:
                log_marg_lhd = tf.constant(0.0)
            loss = -(log_model_density - log_marg_lhd)
        # step optimizer
        grads = tape.gradient(target=loss, sources=self.model.decoder.trainable_weights)
        self.optim.apply_gradients(zip(grads, self.model.decoder.trainable_weights))

        return loss, log_model_density, log_marg_lhd


def create_mlp(
    d_x: int,
    d_y: int,
    n_hidden: int,
    d_hidden: int,
) -> keras.Sequential:
    """
    Generate a standard MLP.
    """

    model = tf.keras.Sequential()
    # TODO: how to treat dynamic number of batch dimensions
    #  here we fix 3 batch dims, 1 data dim
    model.add(keras.layers.Input(shape=(None, None, d_x)))
    for _ in range(n_hidden):
        model.add(keras.layers.Dense(units=d_hidden, activation="relu"))
    model.add(keras.layers.Dense(units=d_y, activation=None))

    return model


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
    initial_cov = prior_scale ** 2 * tf.eye(d_z)  # same as prior covariance

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


def meta_train_np(
    np_model: NP,
    benchmark: MetaLearningBenchmark,
    n_iter: int,
    n_tasks: int,
    gmm_learner_lr_w: float,
    gmm_learner_lr_mu_prec: float,
    gmm_learner_n_samples: int,
    decoder_learner_lr: float,
    decoder_learner_n_samples: int,
    decoder_learner_n_context: Tuple,
    callback=None,
):
    ## create learners
    posterior_learner = PosteriorLearner(
        model=np_model,
        lr_w=gmm_learner_lr_w,
        lr_mu_prec=gmm_learner_lr_mu_prec,
        n_samples=gmm_learner_n_samples,
    )
    decoder_learner = DecoderLearner(
        model=np_model,
        lr=decoder_learner_lr,
        n_samples=decoder_learner_n_samples,
        n_context_min=decoder_learner_n_context[0],
        n_context_max=decoder_learner_n_context[1],
    )

    ## load data
    # TODO: batch dataset?
    # TODO: context-target-split?
    dataset = create_tf_dataset(benchmark)
    dataset = dataset.batch(batch_size=n_tasks)
    for batch in dataset:
        x, y = batch

    ## train model
    np_model.reset_gmm(n_tasks=n_tasks)
    for i in tqdm(range(n_iter)):
        metrics = decoder_learner.step(x=x, y=y)
        posterior_learner.step(x=x, y=y)
        if callback is not None:
            callback(iteration=i, np_model=np_model, metrics=metrics)

    return np_model


def adapt_np(
    np_model: NP,
    benchmark: MetaLearningBenchmark,
    n_iter: int,
    n_tasks: int,
    gmm_learner_lr_w: float,
    gmm_learner_lr_mu_prec: float,
    gmm_learner_n_samples: int,
    callback=None,
):
    # create learners
    posterior_learner = PosteriorLearner(
        model=np_model,
        lr_w=gmm_learner_lr_w,
        lr_mu_prec=gmm_learner_lr_mu_prec,
        n_samples=gmm_learner_n_samples,
    )

    ## load all data
    # TODO: batch dataset?
    # TODO: iterate multiple times over data?
    # TODO: shuffle dataset?
    # TODO: context-target-split?
    dataset = create_tf_dataset(benchmark)
    dataset = dataset.batch(batch_size=n_tasks)
    for batch in dataset:
        x, y = batch

    # train model
    np_model.reset_gmm(n_tasks=n_tasks)
    for i in tqdm(range(n_iter)):
        posterior_learner.step(x=x, y=y)
        if callback is not None:
            callback(iteration=i, np_model=np_model, metrics=None)

    return np_model
