import tensorflow as tf
import tensorflow_probability as tfp
from i_bayes_rule.i_bayesian_learning_rule_gmm import step as gmm_learner_step
from i_bayes_rule.util import GMM
from matplotlib import pyplot as plt
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
        model_prior_scale: float,
        decoder_n_hidden: int,
        decoder_d_hidden: int,
        decoder_output_scale: float,
    ):
        self.d_x = d_x
        self.d_y = d_y
        self.d_z = d_z
        self.gmm_n_components = gmm_n_components
        self.gmm_prior_scale = gmm_prior_scale
        self.model_prior_scale = model_prior_scale
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

    @tf.function
    def predict(self, x: tf.Tensor, z: tf.Tensor):
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

        # check output
        assert mu.shape == (n_samples, self.n_task, n_points, self.d_y)
        return mu

    @tf.function
    def log_likelihood(self, x: tf.Tensor, y: tf.Tensor, z: tf.Tensor):
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
        mu = self.predict(x=x, z=z)
        gaussian = tfp.distributions.Independent(
            tfp.distributions.Normal(loc=mu, scale=self.decoder_output_scale),
            reinterpreted_batch_ndims=1,  # sum ll of data dim upon calling log_prob
        )
        log_likelihood = gaussian.log_prob(y) 
        log_likelihood = tf.reduce_sum(log_likelihood, axis=-1)  # TODO: is this correct? sum ll of datapoints

        # check output
        assert log_likelihood.shape == (n_samples, self.n_task)
        return log_likelihood

    @tf.function
    def log_prior_density(self, z: tf.Tensor):
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
                scale=self.model_prior_scale * tf.ones((self.d_z,)),
            ),
            reinterpreted_batch_ndims=1,  # sum ll of z-dim upon calling log_prob
        )
        log_prior_density = gaussian.log_prob(z)

        # check output
        assert log_prior_density.shape == (n_samples, self.n_task)
        return log_prior_density

    @tf.function
    def log_unnormalized_posterior_density(
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
        log_likelihood = self.log_likelihood(x=x, y=y, z=z)
        assert log_likelihood.shape == (n_samples, self.n_task)
        log_prior_density = self.log_prior_density(z=z)
        assert log_prior_density.shape == (n_samples, self.n_task)
        log_density = log_likelihood + log_prior_density

        # check output
        assert log_density.shape == (n_samples, self.n_task)
        return log_density

    @tf.function
    def log_unnormalized_posterior_density_self_data(self, z: tf.Tensor):
        # This function can be passed to gradient/hessian computations without
        # building a lambda function.
        return self.log_unnormalized_posterior_density(x=self.x, y=self.y, z=z)

    @tf.function
    def log_approximate_posterior_density(self, z: tf.Tensor):
        # check input
        n_samples = z.shape[0]
        n_task = z.shape[1]
        d_z = z.shape[2]

        log_density = self.gmm.log_density(z=z)

        # check output
        assert log_density.shape == (n_samples, n_task)
        return log_density

    # @tf.function
    # TODO: decorating this with tf.function does not allow to change n_tasks
    def sample_approximate_posterior(self, n_samples: int):
        z = self.gmm.sample(n_samples=n_samples)

        # check output
        assert z.shape == (n_samples, self.n_task, self.d_z)
        return z


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
            target_density_fn=self.model.log_unnormalized_posterior_density_self_data,
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
    def step(self, x: tf.Tensor, y: tf.Tensor):
        ## check input
        n_points = x.shape[1]
        assert x.shape == (self.model.n_task, n_points, self.model.d_x)
        assert y.shape == (self.model.n_task, n_points, self.model.d_y)

        ## perform step
        # sample model
        z = self.model.sample_approximate_posterior(n_samples=self.n_samples)
        assert z.shape == (self.n_samples, self.model.n_task, self.model.d_z)
        with tf.GradientTape() as tape:
            # compute elbo = E_q ( -self.model.log_density ) + const in decoder weights
            #  averaged over tasks
            # TODO: prior does not depend on theta yet
            loss = -tf.reduce_mean(
                self.model.log_unnormalized_posterior_density(x=x, y=y, z=z)
            )
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
        model_prior_scale=config["model_prior_scale"],
        decoder_n_hidden=config["decoder_n_hidden"],
        decoder_d_hidden=config["decoder_d_hidden"],
        decoder_output_scale=config["decoder_output_scale"],
    )
    cp_learner = PosteriorLearner(
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

    # TODO: batch dataset?
    # TODO: iterate multiple times over data?
    # TODO: shuffle dataset?
    # TODO: context-target-split?

    # load all data
    dataset_meta = dataset_meta.batch(batch_size=config["n_task_meta"])
    for batch in dataset_meta:
        x, y = batch
    assert x.shape == (
        config["n_task_meta"],
        config["n_datapoints_per_task_meta"],
        config["d_x"],
    )
    assert y.shape == (
        config["n_task_meta"],
        config["n_datapoints_per_task_meta"],
        config["d_y"],
    )

    # train model
    np_model.reset_gmm(n_tasks=config["n_task_meta"])
    for i in tqdm(range(config["n_steps_per_iter"])):
        if callback is not None and i % (config["n_steps_per_iter"] // 5) == 0:
            callback(iteration=i, np_model=np_model, x=x, y=y)
        for _ in range(config["n_theta_steps_per_gmm_step"]):
            th_learner.step(x=x, y=y)
        cp_learner.step(x=x, y=y)

    return np_model


def test_np(config: dict, np_model: NP, dataset_test: tf.data.Dataset):
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
        x, y = batch

    # adapt model
    np_model.reset_gmm(n_tasks=n_task)
    for _ in tqdm(range(config["steps_per_iter"])):
        cp_learner.step(x=x, y=y)

    # make predictions
    n_plt_tasks = config["n_task_test"]
    n_samples = 10
    z = np_model.sample_approximate_posterior(n_samples=n_samples)
    x_plt = tf.linspace(tf.reduce_min(x), tf.reduce_max(x), 128)
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
        ax.scatter(x[i], y[i], label="ground truth")
        ax.grid()
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
    plt.show()
