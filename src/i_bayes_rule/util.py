import tensorflow as tf
import tensorflow_probability as tfp

# TODO: how to assign input_signature with variable number of dimensions?
@tf.function
def expectation_prod_neg(log_a_z: tf.Tensor, b_z: tf.Tensor):
    """
    Compute the expectation E_q[a(z) * b(z)] \approx 1/S \sum_s a(z_s) b(z_s) in a
    stable manner, where z_s is sampled from q, and S is the number of samples.
    b(z_s) may contain negative or zero entries. The 0th dimension is assumed to
    be the sample dimension.
    """
    ## check inputs
    if tf.executing_eagerly():
        n_samples = log_a_z.shape[0]
        assert b_z.shape[0] == n_samples
    else:
        n_samples = tf.shape(log_a_z)[0]

    ## compute expectation
    expectation, signs = tfp.math.reduce_weighted_logsumexp(
        log_a_z + tf.math.log(tf.math.abs(b_z)),
        w=tf.math.sign(b_z),
        axis=0,
        return_sign=True,
    )
    expectation = tf.exp(expectation) * signs
    expectation = 1.0 / tf.cast(n_samples, tf.float32) * expectation

    # check output
    if tf.executing_eagerly():
        assert (
            expectation.shape == tf.broadcast_static_shape(log_a_z.shape, b_z.shape)[1:]
        )
    return expectation


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
    ]
)
def compute_S_bar(
    z: tf.Tensor, loc: tf.Tensor, prec: tf.Tensor, log_tgt_post_grad: tf.Tensor
):
    """
    compute S_bar = prec * (z - mu) * (-log_tgt_post_grad)
    """
    # check input
    if tf.executing_eagerly():
        n_samples = z.shape[0]
        d_z = loc.shape[-1]
        batch_shape = z.shape[1:-1]
        n_components = loc.shape[-2]
        assert z.shape == (n_samples,) + batch_shape + (d_z,)
        assert loc.shape == batch_shape + (n_components, d_z)
        assert prec.shape == batch_shape + (n_components, d_z, d_z)
        assert log_tgt_post_grad.shape == (n_samples,) + batch_shape + (d_z,)

    S_bar = z[..., None, :] - loc[None, ..., :, :]
    # assert S_bar.shape == (n_samples,) + batch_shape + (n_components, d_z)
    S_bar = tf.einsum("...kij,s...kj->s...ki", prec, S_bar)
    # assert S_bar.shape == (n_samples,) + batch_shape + (n_components, d_z)
    S_bar = tf.einsum("s...ki,s...j->s...kij", S_bar, -log_tgt_post_grad)

    # check output
    if tf.executing_eagerly():
        assert S_bar.shape == (n_samples,) + batch_shape + (n_components, d_z, d_z)
    return S_bar


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),
    ]
)
def log_w_to_log_omega(log_w: tf.Tensor):
    """
    Compute log_omega from log_w.
    -> log_omega[k] := log(w[k]/w[K]), k = 1...K-1
    """
    # check input
    if tf.executing_eagerly():
        n_components = log_w.shape[-1]

    # compute log_omega
    log_omega = log_w[..., :-1] - log_w[..., -1:]

    # check output
    if tf.executing_eagerly():
        assert log_omega.shape[-1:] == (n_components - 1,)
    return log_omega


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.float32),
    ]
)
def log_omega_to_log_w(log_omega: tf.Tensor):
    """
    Compute log_w from log_omega.
    -> log_w[k] = log_omega[k]  - logsumexp(log_omega_tilde) for k=1..K-1
       with log_omega_tilde[k] = log_omega[k] (for k=1..K-1)
            log_omega_tilde[k] = 0            (for k=K)
    """
    # check input
    if tf.executing_eagerly():
        n_components = log_omega.shape[-1] + 1

    # compute log_omega_tilde
    log_omega_tilde = tf.concat(
        [log_omega, tf.zeros(tf.concat([tf.shape(log_omega)[:-1], [1]], axis=0))],
        axis=-1,
    )

    # compute log_w
    lse_log_omega_tilde = tf.reduce_logsumexp(log_omega_tilde, keepdims=True, axis=-1)
    log_w = log_omega - lse_log_omega_tilde
    log_w = tf.concat((log_w, -lse_log_omega_tilde), axis=-1)

    # check output
    if tf.executing_eagerly():
        assert log_w.shape[-1:] == (n_components,)
    return log_w

