import tensorflow as tf
import tensorflow_probability as tfp
from i_bayes_rule.i_bayesian_learning_rule_gmm import step as gmm_learner_step
from i_bayes_rule.util import GMM
from matplotlib import pyplot as plt
from tensorflow import keras
from tqdm import tqdm


# TODO: check that all functions decorated with tf.function always are called with the same tensor-shapes!
# TODO: normalize data


class NP:
    """
    p(D^t | D^c, theta) = \int p(D^t | z, theta) p(z | D^c, theta) dz
    """

    def __init__(
        self,
        d_x: int,
        d_y: int,
        d_z: int,
        gmm_n_components: int,
        gmm_prior_scale: float,
        decoder_n_hidden: int,
        decoder_d_hidden: int,
        decoder_output_scale: float,
    ):
        self.d_x = d_x
        self.d_y = d_y
        self.d_z = d_z
        self.gmm_n_components = gmm_n_components
        self.gmm_prior_scale = gmm_prior_scale
        self.decoder_n_hidden = decoder_n_hidden
        self.decoder_d_hidden = decoder_d_hidden
        self.decoder_output_scale = decoder_output_scale

        self.decoder = create_mlp(
            d_x=self.d_x + self.d_z,
            d_y=self.d_y,
            n_hidden=self.decoder_n_hidden,
            d_hidden=self.decoder_d_hidden,
        )
        self.n_tasks = None
        self.gmm = None

    def reset_gmm(self, n_tasks: int):
        # TODO: in order to learn a prior, this has to be deterministic and the
        #  same for each task!
        self.n_tasks = n_tasks
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

    @tf.function
    def predict(self, x: tf.Tensor, z: tf.Tensor):
        # check input
        n_points = x.shape[1]
        n_samples = z.shape[0]
        assert x.shape == (self.n_tasks, n_points, self.d_x)
        assert z.shape == (n_samples, self.n_tasks, self.d_z)

        # TODO: do we need the broadcasting or can this be done automatically by keras?
        x = tf.broadcast_to(x[None, ...], (n_samples, self.n_tasks, n_points, self.d_x))
        z = tf.broadcast_to(
            z[..., None, :], (n_samples, self.n_tasks, n_points, self.d_z)
        )
        xz = tf.concat((x, z), axis=-1)
        assert xz.shape == (n_samples, self.n_tasks, n_points, self.d_x + self.d_z)
        mu = self.decoder(xz)

        # check output
        assert mu.shape == (n_samples, self.n_tasks, n_points, self.d_y)
        return mu

    @tf.function
    def log_likelihood(self, x: tf.Tensor, y: tf.Tensor, z: tf.Tensor):
        """
        log p(D^t | z, theta)
        """
        # check input
        n_points = x.shape[1]
        n_samples = z.shape[0]
        assert x.shape == (self.n_tasks, n_points, self.d_x)
        assert y.shape == (self.n_tasks, n_points, self.d_y)
        assert z.shape == (n_samples, self.n_tasks, self.d_z)

        # compute log likelihood
        y = tf.broadcast_to(y[None, ...], (n_samples, self.n_tasks, n_points, self.d_y))
        mu = self.predict(x=x, z=z)
        gaussian = tfp.distributions.Independent(
            tfp.distributions.Normal(loc=mu, scale=self.decoder_output_scale),
            reinterpreted_batch_ndims=1,  # sum ll of data dim upon calling log_prob
        )
        log_likelihood = gaussian.log_prob(y)
        log_likelihood = tf.reduce_sum(log_likelihood, axis=-1)  # sum ll of datapoints

        # check output
        assert log_likelihood.shape == (n_samples, self.n_tasks)
        return log_likelihood

    @tf.function
    def log_conditional_prior_density(self, z: tf.Tensor):
        """
        log p(z | D^c, theta)
        """
        # check input
        n_samples = z.shape[0]
        assert z.shape == (n_samples, self.n_tasks, self.d_z)

        # compute log conditional prior density
        log_density = self.gmm.log_density(z=z)

        # check output
        assert log_density.shape == (n_samples, self.n_tasks)
        return log_density

    @tf.function
    def log_density(self, x: tf.Tensor, y: tf.Tensor, z: tf.Tensor):
        """
        log p(D^t | z, theta) + log p(z | D^c, theta)
        """
        # check input
        n_points = x.shape[1]
        n_samples = z.shape[0]
        assert x.shape == (self.n_tasks, n_points, self.d_x)
        assert y.shape == (self.n_tasks, n_points, self.d_y)
        assert z.shape == (n_samples, self.n_tasks, self.d_z)

        # compute log_density
        log_likelihood = self.log_likelihood(x=x, y=y, z=z)
        assert log_likelihood.shape == (n_samples, self.n_tasks)
        log_conditional_prior_density = self.log_conditional_prior_density(z=z)
        assert log_conditional_prior_density.shape == (n_samples, self.n_tasks)
        log_density = log_likelihood + log_conditional_prior_density

        # check output
        assert log_density.shape == (n_samples, self.n_tasks)
        return log_density

    @tf.function
    def log_density_self_data(self, z: tf.Tensor):
        # This function can be passed to gradient/hessian computations without
        # building a lambda function.
        return self.log_density(x=self.x_tgt, y=self.y_tgt, z=z)

    # @tf.function
    # TODO: decorating this with tf.function does not allow to change n_tasks
    def sample_z(self, n_samples: int):
        z = self.gmm.sample(n_samples=n_samples)

        # check output
        assert z.shape == (n_samples, self.n_tasks, self.d_z)
        return z


class ConditionalPriorLearner:
    def __init__(self, model: NP, lr_w: float, lr_mu_prec: float, n_samples: int):
        self.model = model
        self.lr_w = lr_w
        self.lr_mu_prec = lr_mu_prec
        self.n_samples = n_samples

    def step(
        self, x_tgt: tf.Tensor, y_tgt: tf.Tensor, x_ctx: tf.Tensor, y_ctx: tf.Tensor
    ):
        ## check input
        n_points_tgt = x_tgt.shape[1]
        n_points_ctx = x_ctx.shape[1]
        assert x_tgt.shape == (self.model.n_tasks, n_points_tgt, self.model.d_x)
        assert y_tgt.shape == (self.model.n_tasks, n_points_tgt, self.model.d_y)
        assert x_ctx.shape == (self.model.n_tasks, n_points_ctx, self.model.d_x)
        assert y_ctx.shape == (self.model.n_tasks, n_points_ctx, self.model.d_y)

        ## step
        # TODO: is there a better way than setting model attributes here?
        self.model.x_tgt = x_tgt
        self.model.y_tgt = y_tgt
        model, _ = gmm_learner_step(
            model=self.model.gmm,
            target_density_fn=self.model.log_density_self_data,
            n_samples=self.n_samples,
            lr_w=self.lr_w,
            lr_mu_prec=self.lr_mu_prec,
            prec_method="hessian",
        )


class LikelihoodLearner:
    def __init__(self, model: NP, lr: float, n_samples: int):
        self.model = model
        self.lr = lr
        self.n_samples = n_samples
        self.optim = keras.optimizers.Adam(learning_rate=self.lr)

    @tf.function
    def step(self, x_tgt: tf.Tensor, y_tgt: tf.Tensor):
        ## check input
        n_points = x_tgt.shape[1]
        assert x_tgt.shape == (self.model.n_tasks, n_points, self.model.d_x)
        assert y_tgt.shape == (self.model.n_tasks, n_points, self.model.d_y)

        ## perform step
        # sample model
        z = self.model.sample_z(n_samples=self.n_samples)
        assert z.shape == (self.n_samples, self.model.n_tasks, self.model.d_z)
        with tf.GradientTape() as tape:
            # compute likelihood
            ll = self.model.log_likelihood(x=x_tgt, y=y_tgt, z=z)
            assert ll.shape == (self.n_samples, self.model.n_tasks)
            # compute loss
            loss = -tf.math.log(tf.cast(self.n_samples, tf.float32))
            loss = loss + tf.math.reduce_logsumexp(ll, axis=0, keepdims=True)
            loss = tf.reduce_sum(loss, axis=1, keepdims=True)
            assert loss.shape == (1, 1)
            loss = tf.squeeze(loss)
        # step optimizer
        grads = tape.gradient(target=loss, sources=self.model.decoder.trainable_weights)
        self.optim.apply_gradients(zip(grads, self.model.decoder.trainable_weights))

        return loss


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
    return weights, means, covs


def shuffle_and_split(batch, n_ctx):
    # TODO: shuffle using tf dataset functionality?
    # check input
    x = batch[0]
    y = batch[1]
    n_task = x.shape[0]
    n_datapoints_per_task = x.shape[1]
    assert n_ctx < n_datapoints_per_task

    # TODO: shuffle datapoints for each task and split into context and target
    # idx = tf.random.shuffle(tf.range(n_datapoints_per_task))
    # x = x[:, idx, :]
    # y = x[:, idx, :]
    x_ctx = x[:, :n_ctx, :]
    y_ctx = y[:, :n_ctx, :]
    x_tgt = x[:, n_ctx:, :]
    y_tgt = y[:, n_ctx:, :]

    return x_ctx, y_ctx, x_tgt, y_tgt


def meta_train_np(config: dict, dataset_meta: tf.data.Dataset, callback=None):
    # create learners
    np_model = NP(
        d_x=config["d_x"],
        d_y=config["d_y"],
        d_z=config["d_z"],
        gmm_n_components=config["gmm_n_components"],
        gmm_prior_scale=config["gmm_prior_scale"],
        decoder_n_hidden=config["decoder_n_hidden"],
        decoder_d_hidden=config["decoder_d_hidden"],
        decoder_output_scale=config["decoder_output_scale"],
    )
    cp_learner = ConditionalPriorLearner(
        model=np_model,
        lr_w=config["gmm_learner_lr_w"],
        lr_mu_prec=config["gmm_learner_lr_mu_prec"],
        n_samples=config["gmm_learner_n_samples"],
    )
    th_learner = LikelihoodLearner(
        model=np_model,
        lr=config["decoder_learner_lr"],
        n_samples=config["decoder_learner_n_samples"],
    )

    # batch dataset
    # TODO: implement shuffling and batching without having to call np_model.reset
    #  -> store all task-GMMs by index
    # meta_dataset = meta_dataset.shuffle(buffer_size=config["batch_size"])
    # TODO: check shuffle
    # batched version not implemented yet
    assert config["batch_size"] == config["n_task_meta"]
    dataset_meta = dataset_meta.batch(config["batch_size"])
    # dataset_meta = dataset_meta.repeat(config["n_iter"]) # not implemented

    # load data
    for batch in dataset_meta:
        x_ctx, y_ctx, x_tgt, y_tgt = shuffle_and_split(
            batch=batch, n_ctx=config["n_context"]
        )
        assert (
            x_tgt.shape[0] == config["n_task_meta"]
        )  # batched version not implemented

    # train model
    np_model.reset_gmm(n_tasks=config["n_task_meta"])
    for _ in tqdm(range(config["steps_per_iter"])):
        cp_learner.step(x_tgt=x_tgt, y_tgt=y_tgt, x_ctx=x_ctx, y_ctx=y_ctx)
        th_learner.step(x_tgt=x_tgt, y_tgt=y_tgt)
        if callback is not None:
            callback(iter=iter, np_model=np_model)

    return np_model


def test_np(config: dict, np_model: NP, dataset_test: tf.data.Dataset, n_context: int):
    cp_learner = ConditionalPriorLearner(
        model=np_model,
        lr_w=config["gmm_learner_lr_w"],
        lr_mu_prec=config["gmm_learner_lr_mu_prec"],
        n_samples=config["gmm_learner_n_samples"],
    )
    n_task = config["n_task_test"]

    # generate some prediction plots
    dataset_test = dataset_test.batch(n_task)

    # load data
    np_model.reset_gmm(n_tasks=n_task)
    for batch in dataset_test:
        x_ctx, y_ctx, x_tgt, y_tgt = shuffle_and_split(batch=batch, n_ctx=n_context)

    # adapt model
    for _ in tqdm(range(config["steps_per_iter"])):
        cp_learner.step(x_tgt=x_tgt, y_tgt=y_tgt, x_ctx=x_ctx, y_ctx=y_ctx)

    # make predictions
    n_plt_tasks = config["n_task_test"]
    n_samples = 10
    z = np_model.sample_z(n_samples=n_samples)
    x_plt = tf.linspace(tf.reduce_min(x_tgt), tf.reduce_max(x_tgt), 128)
    x_plt = tf.reshape(x_plt, (-1, 1))
    x_plt = tf.repeat(x_plt[None, :, :], repeats=n_task, axis=0)
    mu_pred = np_model.predict(x=x_plt, z=z)
    fig, axes = plt.subplots(
        nrows=n_plt_tasks, ncols=1, squeeze=False, sharex=True, sharey=True
    )
    for i in range(n_plt_tasks):
        ax = axes[i, 0]
        for j in range(n_samples):
            ax.plot(x_plt[i], mu_pred[j, i], label="pred")
        ax.scatter(x_tgt[i], y_tgt[i], label="ground truth")
        ax.scatter(x_ctx[i], y_ctx[i], label="context")
        ax.grid()
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
    plt.show()
