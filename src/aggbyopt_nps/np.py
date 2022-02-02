import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp

from i_bayes_rule.util import GMM


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

    def log_likelihood(self, x_t: tf.Tensor, y_t: tf.Tensor, z: tf.Tensor):
        """
        log p(D^t | z, theta)
        """
        # check input
        n_points = x_t.shape[1]
        n_samples = z.shape[0]
        assert x_t.shape == (self.n_tasks, n_points, self.d_x)
        assert y_t.shape == (self.n_tasks, n_points, self.d_y)
        assert z.shape == (n_samples, self.n_tasks, self.d_z)

        # compute log likelihood
        # TODO: do we need the broadcasting or can this be done automatically by keras?
        x_t = tf.broadcast_to(
            x_t[None, ...], (n_samples, self.n_tasks, n_points, self.d_x)
        )
        y_t = tf.broadcast_to(
            y_t[None, ...], (n_samples, self.n_tasks, n_points, self.d_y)
        )
        z = tf.broadcast_to(
            z[..., None, :], (n_samples, self.n_tasks, n_points, self.d_z)
        )
        xz = tf.stack((x_t, z), dim=-1)
        assert xz.shape == (n_samples, self.n_tasks, n_points, self.d_x + self.d_z)
        mu = self.decoder(xz)
        assert mu.shape == (n_samples, self.n_tasks, n_points, self.d_y)
        gaussian = tfp.distributions.Independent(
            tfp.distributions.Normal(loc=mu, scale=self.decoder_output_scale),
            reinterpreted_batch_ndims=1,
        )
        log_likelihood = gaussian.log_prob(y_t)

        # check output
        assert log_likelihood.shape == (n_samples, self.n_tasks)
        return log_likelihood

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

    def log_density(self, x_t: tf.Tensor, y_t: tf.Tensor, z: tf.Tensor):
        """
        log p(D^t | z, theta) + log p(z | D^c, theta)
        """
        # check input
        n_points = x_t.shape[1]
        n_samples = z.shape[0]
        assert x_t.shape == (self.n_tasks, n_points, self.d_x)
        assert y_t.shape == (self.n_tasks, n_points, self.d_y)
        assert z.shape == (n_samples, self.n_tasks, self.d_z)

        # compute log_density
        log_likelihood = self.log_likelihood(x_t=x_t, y_t=y_t, z=z)
        assert log_likelihood.shape == (n_samples, self.n_tasks)
        log_conditional_prior_density = (self.log_conditional_prior_density(z=z),)
        assert log_conditional_prior_density.shape == (n_samples, self.n_tasks)
        log_density = log_likelihood + log_conditional_prior_density

        # check output
        assert log_density.shape == (n_samples, self.n_tasks)
        return log_density

    def sample_z(self, n_samples: int):
        z = self.gmm.sample(n_samples=n_samples)

        # check output
        assert z.shape == (n_samples, self.d_z)
        return z


class ConditionalPriorLearner:
    def __init__(self, model: NP, lr: float, n_samples: int):
        self.model = model
        self.lr = lr
        self.n_samples = n_samples

    def step(self, x_t: tf.Tensor, y_t: tf.Tensor, x_c: tf.Tensor, y_c: tf.Tensor):
        ## check input
        n_points_tgt = x_t.shape[1]
        n_points_ctx = x_c.shape[1]
        assert x_t.shape == (self.model.n_tasks, n_points_tgt, self.model.d_x)
        assert y_t.shape == (self.model.n_tasks, n_points_tgt, self.model.d_y)
        assert x_c.shape == (self.model.n_tasks, n_points_ctx, self.model.d_x)
        assert y_c.shape == (self.model.n_tasks, n_points_ctx, self.model.d_y)

        ## step
        gmm_learner_step(
            model=self.model.gmm,
            target_dist=self.model.log_density,
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

    def step(self, x_t: tf.Tensor, y_t: tf.Tensor):
        ## check input
        n_points = x_t.shape[1]
        assert x_t.shape == (self.model.n_tasks, n_points, self.model.d_x)
        assert y_t.shape == (self.model.n_tasks, n_points, self.model.d_y)

        ## perform step
        # sample model
        z = self.model.sample_z(n=self.n_samples)
        assert z.shape == (self.n_samples, self.model.n_tasks, self.model.d_z)
        with tf.GradientTape() as tape:
            # compute likelihood
            ll = self.model.log_likelihood(x_t=x_t, y_t=y_t, z=z)
            assert ll.shape == (self.n_samples, self.model.n_tasks)
            # compute loss
            loss = -tf.math.log(self.n_samples)
            loss = loss + tf.math.reduce_logsumexp(ll, axis=0, keepdims=True)
            loss = tf.reduce_sum(loss, axis=1, keepdims=True)
            assert loss.shape == (1, 1)
            loss = loss.squeeze()
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
    model.add(keras.layers.Input(shape=(d_x,)))
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
    covs = initial_cov.repeat((n_tasks, n_components, 1, 1))

    # check output
    assert weights.shape == (n_tasks, n_components)
    assert means.shape == (n_tasks, n_components, d_z)
    assert covs.shape == (n_tasks, n_components, d_z, d_z)
    return weights, means, covs


def meta_train_np(meta_dataloader, np_model, n_iter, callback):
    cp_learner = ConditionalPriorLearner(model=np_model)
    th_learner = LikelihoodLearner(model=np_model)
    for iter in range(n_iter):
        for batch in meta_dataloader:
            cp_learner.step(x_t=batch.x_t, y_t=batch.y_t, x_c=batch.x_c, y_c=batch.y_c)
            th_learner.step(x_t=batch.x_t, y_t=batch.y_t, x_c=batch.x_c, y_c=batch.y_c)

        callback(iter=iter, np_model=np_model)
