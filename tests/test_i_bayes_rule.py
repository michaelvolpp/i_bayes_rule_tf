import math

import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from i_bayes_rule.util import (
    compute_S_bar,
    expectation_prod_neg,
    gmm_log_component_densities,
    gmm_log_density,
    gmm_log_responsibilities,
    gmm_log_density_grad_hess,
    log_omega_to_log_w,
    log_w_to_log_omega,
    sample_gmm,
    scale_tril_to_cov,
    cov_to_scale_tril,
    prec_to_prec_tril,
    prec_to_scale_tril,
)
from i_bayes_rule.util_autograd import eval_fn_grad_hess


def test_prec_to_prec_tril():
    tf.config.run_functions_eagerly(True)

    # low D
    L_true = tf.constant(
        [
            [[1.0, 0.0], [2.0, 1.0]],
            [[3.0, 0.0], [-7.0, 3.5]],
            [[math.pi, 0.0], [math.e, 124]],
        ],
    )
    prec = L_true @ tf.linalg.matrix_transpose(L_true)
    L_comp = prec_to_prec_tril(prec=prec)
    assert tf.experimental.numpy.allclose(L_comp, L_true)

    # low D, additional batch-dim
    L_true = tf.constant(
        [
            [
                [[1.0, 0.0], [2.0, 1.0]],
                [[3.0, 0.0], [-7.0, 3.5]],
                [[math.pi, 0.0], [math.e, 124]],
            ],
            [
                [[3.0, 0.0], [-70.0, 3.5]],
                [[math.e, 0.0], [math.pi, 124]],
                [[1.0, 0.0], [2.0, 1.0]],
            ],
        ]
    )
    prec = L_true @ tf.linalg.matrix_transpose(L_true)
    L_comp = prec_to_prec_tril(prec=prec)
    assert tf.experimental.numpy.allclose(L_comp, L_true)

    # high D: requires 64-bit precision to pass test
    tf.random.set_seed(123)
    L_true = tf.stack(
        [
            tf.experimental.numpy.tril(tf.random.uniform([10, 10], dtype=tf.float64)),
            tf.experimental.numpy.tril(tf.random.uniform([10, 10], dtype=tf.float64)),
        ]
    )
    prec = L_true @ tf.linalg.matrix_transpose(L_true)
    L_comp = prec_to_prec_tril(prec=prec)
    assert tf.experimental.numpy.allclose(L_comp, L_true)


def test_prec_to_scale_tril():
    tf.config.run_functions_eagerly(True)

    # low D
    L = tf.constant(
        [
            [[1.0, 0.0], [2.0, 1.0]],
            [[3.0, 0.0], [-7.0, 3.5]],
            [[math.pi, 0.0], [math.e, 124]],
        ]
    )
    cov = L @ tf.linalg.matrix_transpose(L)
    prec = tf.linalg.inv(cov)
    scale_tril = prec_to_scale_tril(prec=prec)
    assert tf.experimental.numpy.allclose(scale_tril, L)

    # low D, additional batch-dim
    L_true = tf.constant(
        [
            [
                [[1.0, 0.0], [2.0, 1.0]],
                [[3.0, 0.0], [-7.0, 3.5]],
                [[math.pi, 0.0], [math.e, 124]],
            ],
            [
                [[3.0, 0.0], [-70.0, 3.5]],
                [[math.e, 0.0], [math.pi, 124]],
                [[1.0, 0.0], [2.0, 1.0]],
            ],
        ]
    )
    prec = L_true @ tf.linalg.matrix_transpose(L_true)
    L_comp = prec_to_prec_tril(prec=prec)
    assert tf.experimental.numpy.allclose(L_comp, L_true)

    # high D: requires 64-bit precision to pass test
    tf.random.set_seed(123)
    L = tf.stack(
        [
            tf.experimental.numpy.tril(tf.random.uniform([10, 10], dtype=tf.float64)),
            tf.experimental.numpy.tril(tf.random.uniform([10, 10], dtype=tf.float64)),
        ]
    )
    cov = L @ tf.linalg.matrix_transpose(L)
    prec = tf.linalg.inv(cov)
    scale_tril = prec_to_scale_tril(prec=prec)
    assert tf.experimental.numpy.allclose(scale_tril, L)


def test_scale_tril_to_cov():
    tf.config.run_functions_eagerly(True)

    # check 1
    scale_tril = tf.constant(
        [
            [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
            [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
            [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
        ]
    )
    true_cov = tf.constant(
        [
            [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
            [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
            [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
        ]
    )
    cov = scale_tril_to_cov(scale_tril=scale_tril)
    assert tf.experimental.numpy.allclose(cov, true_cov)

    # check 2 with additional batch dim
    scale_tril = tf.constant(
        [
            [
                [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
                [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
                [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
            ],
            [
                [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
                [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
                [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
            ],
        ]
    )
    true_cov = tf.constant(
        [
            [
                [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
                [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
                [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
            ],
            [
                [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
                [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
                [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
            ],
        ]
    )
    cov = scale_tril_to_cov(scale_tril=scale_tril)
    assert tf.experimental.numpy.allclose(cov, true_cov)


def test_cov_to_scale_tril():
    tf.config.run_functions_eagerly(True)
    
    # check 1
    cov = tf.constant(
        [
            [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
            [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
            [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
        ]
    )
    true_scale_tril = tf.constant(
        [
            [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
            [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
            [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
        ]
    )
    scale_tril = cov_to_scale_tril(cov=cov)
    assert tf.experimental.numpy.allclose(scale_tril, true_scale_tril)

    # check 2: with additional batch dim
    cov = tf.constant(
        [
            [
                [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
                [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
                [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
            ],
            [
                [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
                [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
                [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]],
            ],
        ]
    )
    true_scale_tril = tf.constant(
        [
            [
                [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
                [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
                [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
            ],
            [
                [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
                [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
                [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]],
            ],
        ]
    )
    scale_tril = cov_to_scale_tril(cov=cov)
    assert tf.experimental.numpy.allclose(scale_tril, true_scale_tril)


def test_sample_gmm():
    tf.config.run_functions_eagerly(True)

    # check 0: d_z == 1, n_components == 1
    n_samples = 10
    d_z = 1
    log_w = tf.math.log([1.0])
    loc = tf.constant([[-1.0]])
    scale_tril = tf.constant([[[0.1]]])
    samples = sample_gmm(n_samples=10, log_w=log_w, loc=loc, scale_tril=scale_tril)
    assert samples.shape == (n_samples, d_z)

    # check 1: d_z == 1, n_components == 2
    n_samples = 10
    d_z = 1
    log_w = tf.math.log([0.8, 0.2])
    loc = tf.constant([[-1.0], [1.0]])
    scale_tril = tf.constant([[[0.2]], [[0.2]]])
    samples = sample_gmm(n_samples=10, log_w=log_w, loc=loc, scale_tril=scale_tril)
    assert samples.shape == (n_samples, d_z)

    # check 2: d_z == 2, n_components == 1
    n_samples = 10
    d_z = 2
    log_w = tf.math.log([1.0])
    loc = tf.constant([[-1.0, 1.0]])
    scale_tril = tf.constant([[[0.1, 0.0], [-0.2, 1.0]]])
    samples = sample_gmm(n_samples=10, log_w=log_w, loc=loc, scale_tril=scale_tril)
    assert samples.shape == (n_samples, d_z)

    # check 3: d_z == 2, n_components == 2
    n_samples = 10
    d_z = 2
    n_batch = 3
    log_w = tf.math.log([0.8, 0.2])
    loc = tf.constant(
        [
            [1.0, 1.0],
            [-1.0, 1.0],
        ]
    )
    scale_tril = tf.constant(
        [
            [[0.1, 0.0], [-0.2, 1.0]],
            [[0.2, 0.0], [0.2, 2.0]],
        ],
    )
    samples = sample_gmm(n_samples=10, log_w=log_w, loc=loc, scale_tril=scale_tril)
    assert samples.shape == (n_samples, d_z)

    # check 4: d_z == 2, n_components == 2, batch_dim
    d_z = 2
    n_samples = 10
    n_batch = 3
    log_w = tf.math.log(
        [
            [0.8, 0.2],
            [0.3, 0.7],
            [0.1, 0.9],
        ]
    )
    loc = tf.constant(
        [
            [
                [1.0, 1.0],
                [-1.0, 1.0],
            ],
            [
                [2.0, 2.0],
                [-1.0, 3.0],
            ],
            [
                [2.0, 2.0],
                [-1.0, 3.0],
            ],
        ]
    )
    scale_tril = tf.constant(
        [
            [
                [[0.1, 0.0], [-0.2, 1.0]],
                [[0.2, 0.0], [0.2, 2.0]],
            ],
            [
                [[0.2, 0.0], [-0.3, 2.0]],
                [[0.1, 0.0], [0.5, 3.0]],
            ],
            [
                [[0.2, 0.0], [-0.3, 2.0]],
                [[0.1, 0.0], [0.5, 3.0]],
            ],
        ]
    )
    samples = sample_gmm(n_samples=10, log_w=log_w, loc=loc, scale_tril=scale_tril)
    assert samples.shape == (n_samples, n_batch, d_z)


def test_gmm_log_density():
    tf.config.run_functions_eagerly(True)

    # check 0: d_z == 1, n_components == 1
    n_samples = 10
    z = tf.random.normal((n_samples, 1))
    log_w = tf.math.log([1.0])
    loc = tf.constant([[1.0]])
    scale_tril = tf.constant([[[0.1]]])
    log_densities = gmm_log_density(
        z=z,
        log_w=log_w,
        log_component_densities=gmm_log_component_densities(
            z=z, loc=loc, scale_tril=scale_tril
        ),
    )
    true_log_densities = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(
            logits=log_w,
            # validate_args=True,  # TODO: does not work if n_components == 1
        ),
        components_distribution=tfp.distributions.MultivariateNormalTriL(
            loc=loc, scale_tril=scale_tril, validate_args=True
        ),
        validate_args=True,
    ).log_prob(z)
    assert true_log_densities.shape == (n_samples,)
    assert log_densities.shape == (n_samples,)
    assert tf.experimental.numpy.allclose(log_densities, true_log_densities)

    # check 1: d_z == 1, n_components == 2
    n_samples = 10
    z = tf.random.normal((n_samples, 1))
    log_w = tf.math.log([0.8, 0.2])
    loc = tf.constant(
        [
            [1.0],
            [-1.0],
        ]
    )
    scale_tril = tf.constant(
        [
            [[0.1]],
            [[0.2]],
        ]
    )
    log_densities = gmm_log_density(
        z=z,
        log_w=log_w,
        log_component_densities=gmm_log_component_densities(
            z=z, loc=loc, scale_tril=scale_tril
        ),
    )
    true_log_densities = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(
            logits=log_w,
            validate_args=True,
        ),
        components_distribution=tfp.distributions.MultivariateNormalTriL(
            loc=loc, scale_tril=scale_tril, validate_args=True
        ),
        validate_args=True,
    ).log_prob(z)
    assert true_log_densities.shape == (n_samples,)
    assert log_densities.shape == (n_samples,)
    assert tf.experimental.numpy.allclose(log_densities, true_log_densities)

    # check 2: d_z == 2, n_components == 1
    n_samples = 10
    z = tf.random.normal((n_samples, 2))
    log_w = tf.math.log([1.0])
    loc = tf.constant([[1.0, 1.0]])
    scale_tril = tf.constant([[[0.1, 0.0], [-0.2, 1.0]]])
    log_densities = gmm_log_density(
        z=z,
        log_w=log_w,
        log_component_densities=gmm_log_component_densities(
            z=z, loc=loc, scale_tril=scale_tril
        ),
    )
    true_log_densities = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(
            logits=log_w,
            # validate_args=True,  # TODO: does not work if n_components == 1
        ),
        components_distribution=tfp.distributions.MultivariateNormalTriL(
            loc=loc, scale_tril=scale_tril, validate_args=True
        ),
        validate_args=True,
    ).log_prob(z)
    assert true_log_densities.shape == (n_samples,)
    assert log_densities.shape == (n_samples,)
    assert tf.experimental.numpy.allclose(log_densities, true_log_densities)

    # check 3: d_z == 2, n_components == 2
    n_samples = 10
    z = tf.random.normal((n_samples, 2))
    log_w = tf.math.log([0.8, 0.2])
    loc = tf.constant(
        [
            [1.0, 1.0],
            [-1.0, 1.0],
        ]
    )
    scale_tril = tf.constant(
        [
            [[0.1, 0.0], [-0.2, 1.0]],
            [[0.2, 0.0], [0.2, 2.0]],
        ]
    )
    log_densities = gmm_log_density(
        z=z,
        log_w=log_w,
        log_component_densities=gmm_log_component_densities(
            z=z, loc=loc, scale_tril=scale_tril
        ),
    )
    true_log_densities = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(
            logits=log_w,
            validate_args=True,
        ),
        components_distribution=tfp.distributions.MultivariateNormalTriL(
            loc=loc, scale_tril=scale_tril, validate_args=True
        ),
        validate_args=True,
    ).log_prob(z)
    assert true_log_densities.shape == (n_samples,)
    assert log_densities.shape == (n_samples,)
    assert tf.experimental.numpy.allclose(log_densities, true_log_densities)

    # check 4: d_z == 2, n_components == 2, batch_dim
    n_samples = 10
    n_batch = 3
    z = tf.random.normal((n_samples, n_batch, 2))
    log_w = tf.math.log(
        [
            [0.8, 0.2],
            [0.3, 0.7],
            [0.1, 0.9],
        ]
    )
    loc = tf.constant(
        [
            [
                [1.0, 1.0],
                [-1.0, 1.0],
            ],
            [
                [2.0, 2.0],
                [-1.0, 3.0],
            ],
            [
                [2.0, 2.0],
                [-1.0, 3.0],
            ],
        ]
    )
    scale_tril = tf.constant(
        [
            [
                [[0.1, 0.0], [-0.2, 1.0]],
                [[0.2, 0.0], [0.2, 2.0]],
            ],
            [
                [[0.2, 0.0], [-0.3, 2.0]],
                [[0.1, 0.0], [0.5, 3.0]],
            ],
            [
                [[0.2, 0.0], [-0.3, 2.0]],
                [[0.1, 0.0], [0.5, 3.0]],
            ],
        ]
    )
    log_densities = gmm_log_density(
        z=z,
        log_w=log_w,
        log_component_densities=gmm_log_component_densities(
            z=z, loc=loc, scale_tril=scale_tril
        ),
    )
    true_log_densities = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(
            logits=log_w,
            validate_args=True,
        ),
        components_distribution=tfp.distributions.MultivariateNormalTriL(
            loc=loc, scale_tril=scale_tril, validate_args=True
        ),
        validate_args=True,
    ).log_prob(z)
    assert true_log_densities.shape == (n_samples, n_batch)
    assert log_densities.shape == (n_samples, n_batch)
    assert tf.experimental.numpy.allclose(log_densities, true_log_densities)


def test_gmm_log_component_densities():
    tf.config.run_functions_eagerly(True)
    
    # check 1: d_z == 1, n_components == 1
    n_samples = 10
    n_components = 1
    z = tf.random.normal((n_samples, 1))
    loc = tf.constant([[1.0]])
    scale_tril = tf.constant([[[0.1]]])
    log_component_densities = gmm_log_component_densities(
        z=z,
        loc=loc,
        scale_tril=scale_tril,
    )
    true_log_component_densities = tfp.distributions.MultivariateNormalTriL(
        loc=loc, scale_tril=scale_tril, validate_args=True
    ).log_prob(z[:, None, :])
    assert true_log_component_densities.shape == (n_samples, n_components)
    assert log_component_densities.shape == (n_samples, n_components)
    assert tf.experimental.numpy.allclose(
        true_log_component_densities, log_component_densities
    )

    # check 2: d_z == 2, n_components == 1
    n_samples = 10
    n_components = 1
    z = tf.random.normal((n_samples, 2))
    loc = tf.constant([[1.0, -1.0]])
    scale_tril = tf.constant([[[0.1, 0.0], [-2.0, 1.0]]])
    log_component_densities = gmm_log_component_densities(
        z=z,
        loc=loc,
        scale_tril=scale_tril,
    )
    true_log_component_densities = tfp.distributions.MultivariateNormalTriL(
        loc=loc, scale_tril=scale_tril, validate_args=True
    ).log_prob(z[:, None, :])
    assert true_log_component_densities.shape == (n_samples, n_components)
    assert log_component_densities.shape == (n_samples, n_components)
    assert tf.experimental.numpy.allclose(
        true_log_component_densities, log_component_densities
    )

    # check 3: d_z == 2, n_components == 2
    n_samples = 10
    n_components = 2
    z = tf.random.normal((n_samples, 2))
    loc = tf.constant([[1.0, -1.0], [1.0, 1.0]])
    scale_tril = tf.constant([[[0.1, 0.0], [-2.0, 1.0]], [[0.2, 0.0], [-2.0, 1.0]]])
    log_component_densities = gmm_log_component_densities(
        z=z,
        loc=loc,
        scale_tril=scale_tril,
    )
    true_log_component_densities = tfp.distributions.MultivariateNormalTriL(
        loc=loc, scale_tril=scale_tril, validate_args=True
    ).log_prob(z[:, None, :])
    assert true_log_component_densities.shape == (n_samples, n_components)
    assert log_component_densities.shape == (n_samples, n_components)
    assert tf.experimental.numpy.allclose(
        true_log_component_densities, log_component_densities
    )

    # check 4: d_z == 2, n_components == 2, batch_dim
    n_samples = 10
    n_components = 2
    n_batch = 2
    z = tf.random.normal((n_samples, 2, 2))
    loc = tf.constant(
        [
            [[1.0, -1.0], [1.0, 1.0]],
            [[2.0, -2.0], [2.0, 2.0]],
        ]
    )
    scale_tril = tf.constant(
        [
            [[[0.1, 0.0], [-2.0, 1.0]], [[0.2, 0.0], [-2.0, 1.0]]],
            [[[0.3, 0.0], [-3.0, 2.0]], [[0.4, 0.0], [-1.0, 2.0]]],
        ]
    )
    log_component_densities = gmm_log_component_densities(
        z=z,
        loc=loc,
        scale_tril=scale_tril,
    )
    true_log_component_densities = tfp.distributions.MultivariateNormalTriL(
        loc=loc, scale_tril=scale_tril, validate_args=True
    ).log_prob(z[:, :, None, :])
    assert true_log_component_densities.shape == (n_samples, n_batch, n_components)
    assert log_component_densities.shape == (n_samples, n_batch, n_components)
    assert tf.experimental.numpy.allclose(
        true_log_component_densities, log_component_densities
    )


def test_gmm_log_responsibilities():
    tf.config.run_functions_eagerly(True)

    # check 0: d_z == 1, n_components == 1
    n_samples = 10
    n_components = 1
    z = tf.random.normal((n_samples, 1))
    log_w = tf.math.log([1.0])
    loc = tf.constant([[1.0]])
    scale_tril = tf.constant([[[0.1]]])
    log_component_densities = gmm_log_component_densities(
        z=z, loc=loc, scale_tril=scale_tril
    )
    log_density = gmm_log_density(
        z=z, log_w=log_w, log_component_densities=log_component_densities
    )
    log_responsibilities = gmm_log_responsibilities(
        z=z,
        log_w=log_w,
        log_component_densities=log_component_densities,
        log_density=log_density,
    )
    true_log_responsibilities = tf.math.log_softmax(
        tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(
                logits=log_w,
                # validate_args=True,  # TODO: does not work if n_components == 1
            ),
            components_distribution=tfp.distributions.MultivariateNormalTriL(
                loc=loc, scale_tril=scale_tril, validate_args=True
            ),
            validate_args=True,
        )
        .posterior_marginal(z)
        .logits
    )
    assert true_log_responsibilities.shape == (n_samples, n_components)
    assert log_responsibilities.shape == (n_samples, n_components)
    assert tf.experimental.numpy.allclose(
        tf.math.exp(log_responsibilities),
        tf.math.exp(true_log_responsibilities),
    )

    # check 1: d_z == 1, n_components == 2
    n_samples = 10
    n_components = 2
    z = tf.random.normal((n_samples, 1))
    log_w = tf.math.log([0.8, 0.2])
    loc = tf.constant(
        [
            [1.0],
            [-1.0],
        ]
    )
    scale_tril = tf.constant(
        [
            [[0.1]],
            [[0.2]],
        ]
    )
    log_component_densities = gmm_log_component_densities(
        z=z, loc=loc, scale_tril=scale_tril
    )
    log_density = gmm_log_density(
        z=z, log_w=log_w, log_component_densities=log_component_densities
    )
    log_responsibilities = gmm_log_responsibilities(
        z=z,
        log_w=log_w,
        log_component_densities=log_component_densities,
        log_density=log_density,
    )
    true_log_responsibilities = tf.math.log_softmax(
        tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(
                logits=log_w,
                validate_args=True,
            ),
            components_distribution=tfp.distributions.MultivariateNormalTriL(
                loc=loc, scale_tril=scale_tril, validate_args=True
            ),
            validate_args=True,
        )
        .posterior_marginal(z)
        .logits
    )
    assert true_log_responsibilities.shape == (n_samples, n_components)
    assert log_responsibilities.shape == (n_samples, n_components)
    assert tf.experimental.numpy.allclose(
        tf.math.exp(log_responsibilities),
        tf.math.exp(true_log_responsibilities),
    )

    # check 2: d_z == 2, n_components == 1
    n_samples = 10
    n_components = 1
    z = tf.random.normal((n_samples, 2))
    log_w = tf.math.log([1.0])
    loc = tf.constant([[1.0, 1.0]])
    scale_tril = tf.constant([[[0.1, 0.0], [-0.2, 1.0]]])
    log_component_densities = gmm_log_component_densities(
        z=z, loc=loc, scale_tril=scale_tril
    )
    log_density = gmm_log_density(
        z=z, log_w=log_w, log_component_densities=log_component_densities
    )
    log_responsibilities = gmm_log_responsibilities(
        z=z,
        log_w=log_w,
        log_component_densities=log_component_densities,
        log_density=log_density,
    )
    true_log_responsibilities = tf.math.log_softmax(
        tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(
                logits=log_w,
                # validate_args=True,  # TODO: does not work if n_components == 1
            ),
            components_distribution=tfp.distributions.MultivariateNormalTriL(
                loc=loc, scale_tril=scale_tril, validate_args=True
            ),
            validate_args=True,
        )
        .posterior_marginal(z)
        .logits
    )
    assert true_log_responsibilities.shape == (n_samples, n_components)
    assert log_responsibilities.shape == (n_samples, n_components)
    assert tf.experimental.numpy.allclose(
        tf.math.exp(log_responsibilities),
        tf.math.exp(true_log_responsibilities),
    )

    # check 3: d_z == 2, n_components == 2
    n_samples = 10
    n_components = 2
    z = tf.random.normal((n_samples, 2))
    log_w = tf.math.log([0.8, 0.2])
    loc = tf.constant(
        [
            [1.0, 1.0],
            [-1.0, 1.0],
        ]
    )
    scale_tril = tf.constant(
        [
            [[0.1, 0.0], [-0.2, 1.0]],
            [[0.2, 0.0], [0.2, 2.0]],
        ]
    )
    log_component_densities = gmm_log_component_densities(
        z=z, loc=loc, scale_tril=scale_tril
    )
    log_density = gmm_log_density(
        z=z, log_w=log_w, log_component_densities=log_component_densities
    )
    log_responsibilities = gmm_log_responsibilities(
        z=z,
        log_w=log_w,
        log_component_densities=log_component_densities,
        log_density=log_density,
    )
    true_log_responsibilities = tf.math.log_softmax(
        tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(
                logits=log_w,
                # validate_args=True,  # TODO: does not work if n_components == 1
            ),
            components_distribution=tfp.distributions.MultivariateNormalTriL(
                loc=loc, scale_tril=scale_tril, validate_args=True
            ),
            validate_args=True,
        )
        .posterior_marginal(z)
        .logits
    )
    assert true_log_responsibilities.shape == (n_samples, n_components)
    assert log_responsibilities.shape == (n_samples, n_components)
    assert tf.experimental.numpy.allclose(
        tf.math.exp(log_responsibilities),
        tf.math.exp(true_log_responsibilities),
    )

    # check 4: d_z == 2, n_components == 2, batch_dim
    n_samples = 10
    n_components = 2
    n_batch = 3
    z = tf.random.normal((n_samples, n_batch, 2))
    log_w = tf.math.log(
        [
            [0.8, 0.2],
            [0.3, 0.7],
            [0.1, 0.9],
        ]
    )
    loc = tf.constant(
        [
            [
                [1.0, 1.0],
                [-1.0, 1.0],
            ],
            [
                [2.0, 2.0],
                [-1.0, 3.0],
            ],
            [
                [2.0, 2.0],
                [-1.0, 3.0],
            ],
        ]
    )
    scale_tril = tf.constant(
        [
            [
                [[0.1, 0.0], [-0.2, 1.0]],
                [[0.2, 0.0], [0.2, 2.0]],
            ],
            [
                [[0.2, 0.0], [-0.3, 2.0]],
                [[0.1, 0.0], [0.5, 3.0]],
            ],
            [
                [[0.2, 0.0], [-0.3, 2.0]],
                [[0.1, 0.0], [0.5, 3.0]],
            ],
        ]
    )
    log_component_densities = gmm_log_component_densities(
        z=z, loc=loc, scale_tril=scale_tril
    )
    log_density = gmm_log_density(
        z=z, log_w=log_w, log_component_densities=log_component_densities
    )
    log_responsibilities = gmm_log_responsibilities(
        z=z,
        log_w=log_w,
        log_component_densities=log_component_densities,
        log_density=log_density,
    )
    true_log_responsibilities = tf.math.log_softmax(
        tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(
                logits=log_w,
                # validate_args=True,  # TODO: does not work if n_components == 1
            ),
            components_distribution=tfp.distributions.MultivariateNormalTriL(
                loc=loc, scale_tril=scale_tril, validate_args=True
            ),
            validate_args=True,
        )
        .posterior_marginal(z)
        .logits
    )
    assert true_log_responsibilities.shape == (n_samples, n_batch, n_components)
    assert log_responsibilities.shape == (n_samples, n_batch, n_components)
    assert tf.experimental.numpy.allclose(
        tf.math.exp(log_responsibilities),
        tf.math.exp(true_log_responsibilities),
    )


def test_gmm_log_density_grad_hess():
    tf.config.run_functions_eagerly(True)
    
    # check 0: d_z == 1, n_components == 1
    n_samples = 10
    d_z = 1
    z = tf.random.normal((n_samples, d_z))
    log_w = tf.math.log([1.0])
    loc = tf.constant([[1.0]])
    scale_tril = tf.constant([[[0.1]]])
    log_density, log_density_grad, log_density_hess = gmm_log_density_grad_hess(
        z=z,
        log_w=log_w,
        loc=loc,
        prec=tf.linalg.inv(scale_tril_to_cov(scale_tril)),
        scale_tril=scale_tril,
        compute_grad=True,
        compute_hess=True,
    )
    gmm = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(
            logits=log_w,
            # validate_args=True,  # TODO: does not work if n_components == 1
        ),
        components_distribution=tfp.distributions.MultivariateNormalTriL(
            loc=loc, scale_tril=scale_tril, validate_args=True
        ),
        validate_args=True,
    )
    true_log_density, true_log_density_grad, true_log_density_hess = eval_fn_grad_hess(
        fn=gmm.log_prob,
        z=z,
        compute_grad=True,
        compute_hess=True,
    )
    assert log_density.shape == (n_samples,)
    assert log_density_grad.shape == (n_samples, d_z)
    assert log_density_hess.shape == (n_samples, d_z, d_z)
    assert true_log_density.shape == (n_samples,)
    assert true_log_density_grad.shape == (n_samples, d_z)
    assert true_log_density_hess.shape == (n_samples, d_z, d_z)
    assert tf.experimental.numpy.allclose(true_log_density, log_density)
    assert tf.experimental.numpy.allclose(true_log_density_grad, log_density_grad)
    assert tf.experimental.numpy.allclose(
        true_log_density_hess, log_density_hess, rtol=1e-3
    )

    # check 1: d_z == 1, n_components == 2
    n_samples = 10
    d_z = 1
    z = tf.random.normal((n_samples, d_z))
    log_w = tf.math.log([0.8, 0.2])
    loc = tf.constant(
        [
            [1.0],
            [-1.0],
        ]
    )
    scale_tril = tf.constant(
        [
            [[0.1]],
            [[0.2]],
        ]
    )
    log_density, log_density_grad, log_density_hess = gmm_log_density_grad_hess(
        z=z,
        log_w=log_w,
        loc=loc,
        prec=tf.linalg.inv(scale_tril_to_cov(scale_tril)),
        scale_tril=scale_tril,
        compute_grad=True,
        compute_hess=True,
    )
    gmm = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(
            logits=log_w,
            # validate_args=True,  # TODO: does not work if n_components == 1
        ),
        components_distribution=tfp.distributions.MultivariateNormalTriL(
            loc=loc, scale_tril=scale_tril, validate_args=True
        ),
        validate_args=True,
    )
    true_log_density, true_log_density_grad, true_log_density_hess = eval_fn_grad_hess(
        fn=gmm.log_prob,
        z=z,
        compute_grad=True,
        compute_hess=True,
    )
    assert log_density.shape == (n_samples,)
    assert log_density_grad.shape == (n_samples, d_z)
    assert log_density_hess.shape == (n_samples, d_z, d_z)
    assert true_log_density.shape == (n_samples,)
    assert true_log_density_grad.shape == (n_samples, d_z)
    assert true_log_density_hess.shape == (n_samples, d_z, d_z)
    assert tf.experimental.numpy.allclose(true_log_density, log_density)
    assert tf.experimental.numpy.allclose(true_log_density_grad, log_density_grad)
    assert tf.experimental.numpy.allclose(
        true_log_density_hess, log_density_hess, rtol=1e-3
    )

    # check 2: d_z == 2, n_components == 1
    n_samples = 10
    d_z = 2
    z = tf.random.normal((n_samples, d_z))
    log_w = tf.math.log([1.0])
    loc = tf.constant([[1.0, 1.0]])
    scale_tril = tf.constant([[[0.1, 0.0], [-0.2, 1.0]]])
    log_density, log_density_grad, log_density_hess = gmm_log_density_grad_hess(
        z=z,
        log_w=log_w,
        loc=loc,
        prec=tf.linalg.inv(scale_tril_to_cov(scale_tril)),
        scale_tril=scale_tril,
        compute_grad=True,
        compute_hess=True,
    )
    gmm = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(
            logits=log_w,
            # validate_args=True,  # TODO: does not work if n_components == 1
        ),
        components_distribution=tfp.distributions.MultivariateNormalTriL(
            loc=loc, scale_tril=scale_tril, validate_args=True
        ),
        validate_args=True,
    )
    true_log_density, true_log_density_grad, true_log_density_hess = eval_fn_grad_hess(
        fn=gmm.log_prob,
        z=z,
        compute_grad=True,
        compute_hess=True,
    )
    assert log_density.shape == (n_samples,)
    assert log_density_grad.shape == (n_samples, d_z)
    assert log_density_hess.shape == (n_samples, d_z, d_z)
    assert true_log_density.shape == (n_samples,)
    assert true_log_density_grad.shape == (n_samples, d_z)
    assert true_log_density_hess.shape == (n_samples, d_z, d_z)
    assert tf.experimental.numpy.allclose(true_log_density, log_density)
    assert tf.experimental.numpy.allclose(true_log_density_grad, log_density_grad)
    assert tf.experimental.numpy.allclose(
        true_log_density_hess, log_density_hess, rtol=1e-3
    )

    # check 3: d_z == 2, n_components == 2
    n_samples = 10
    d_z = 2
    n_components = 2
    z = tf.random.normal((n_samples, d_z))
    log_w = tf.math.log([0.8, 0.2])
    loc = tf.constant(
        [
            [1.0, 1.0],
            [-1.0, 1.0],
        ]
    )
    scale_tril = tf.constant(
        [
            [[0.1, 0.0], [-0.2, 1.0]],
            [[0.2, 0.0], [0.2, 2.0]],
        ]
    )
    log_density, log_density_grad, log_density_hess = gmm_log_density_grad_hess(
        z=z,
        log_w=log_w,
        loc=loc,
        prec=tf.linalg.inv(scale_tril_to_cov(scale_tril)),
        scale_tril=scale_tril,
        compute_grad=True,
        compute_hess=True,
    )
    gmm = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(
            logits=log_w,
            # validate_args=True,  # TODO: does not work if n_components == 1
        ),
        components_distribution=tfp.distributions.MultivariateNormalTriL(
            loc=loc, scale_tril=scale_tril, validate_args=True
        ),
        validate_args=True,
    )
    true_log_density, true_log_density_grad, true_log_density_hess = eval_fn_grad_hess(
        fn=gmm.log_prob,
        z=z,
        compute_grad=True,
        compute_hess=True,
    )
    assert log_density.shape == (n_samples,)
    assert log_density_grad.shape == (n_samples, d_z)
    assert log_density_hess.shape == (n_samples, d_z, d_z)
    assert true_log_density.shape == (n_samples,)
    assert true_log_density_grad.shape == (n_samples, d_z)
    assert true_log_density_hess.shape == (n_samples, d_z, d_z)
    assert tf.experimental.numpy.allclose(true_log_density, log_density)
    assert tf.experimental.numpy.allclose(true_log_density_grad, log_density_grad)
    assert tf.experimental.numpy.allclose(
        true_log_density_hess, log_density_hess, rtol=1e-3
    )

    # check 4: d_z == 2, n_components == 2, batch_dim
    n_samples = 10
    n_components = 2
    d_z = 2
    n_batch = 3
    z = tf.random.normal((n_samples, n_batch, d_z))
    log_w = tf.math.log(
        [
            [0.8, 0.2],
            [0.3, 0.7],
            [0.1, 0.9],
        ]
    )
    loc = tf.constant(
        [
            [
                [1.0, 1.0],
                [-1.0, 1.0],
            ],
            [
                [2.0, 2.0],
                [-1.0, 3.0],
            ],
            [
                [2.0, 2.0],
                [-1.0, 3.0],
            ],
        ]
    )
    scale_tril = tf.constant(
        [
            [
                [[0.1, 0.0], [-0.2, 1.0]],
                [[0.2, 0.0], [0.2, 2.0]],
            ],
            [
                [[0.2, 0.0], [-0.3, 2.0]],
                [[0.1, 0.0], [0.5, 3.0]],
            ],
            [
                [[0.2, 0.0], [-0.3, 2.0]],
                [[0.1, 0.0], [0.5, 3.0]],
            ],
        ]
    )
    log_density, log_density_grad, log_density_hess = gmm_log_density_grad_hess(
        z=z,
        log_w=log_w,
        loc=loc,
        prec=tf.linalg.inv(scale_tril_to_cov(scale_tril)),
        scale_tril=scale_tril,
        compute_grad=True,
        compute_hess=True,
    )
    gmm = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(
            logits=log_w,
            # validate_args=True,  # TODO: does not work if n_components == 1
        ),
        components_distribution=tfp.distributions.MultivariateNormalTriL(
            loc=loc, scale_tril=scale_tril, validate_args=True
        ),
        validate_args=True,
    )
    true_log_density, true_log_density_grad, true_log_density_hess = eval_fn_grad_hess(
        fn=gmm.log_prob,
        z=z,
        compute_grad=True,
        compute_hess=True,
    )
    assert log_density.shape == (n_samples, n_batch)
    assert log_density_grad.shape == (n_samples, n_batch, d_z)
    assert log_density_hess.shape == (n_samples, n_batch, d_z, d_z)
    assert true_log_density.shape == (n_samples, n_batch)
    assert true_log_density_grad.shape == (n_samples, n_batch, d_z)
    assert true_log_density_hess.shape == (n_samples, n_batch, d_z, d_z)
    assert tf.experimental.numpy.allclose(true_log_density, log_density)
    assert tf.experimental.numpy.allclose(true_log_density_grad, log_density_grad)
    assert tf.experimental.numpy.allclose(
        true_log_density_hess, log_density_hess, rtol=1e-3
    )


def test_expectation_prod_neg():
    tf.config.run_functions_eagerly(True)

    # check 1
    a_z = tf.constant(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [5.0, 6.0],
            [5.0, 6.0],
        ]
    )
    b_z = tf.constant(
        [
            [3.0, 1.0, -3.0],
            [-2.0, 1.0, 0.0],
            [1.0, 2.0, -17.0],
            [0.0, 0.0, 0.0],
            [-190.0, -200.0, -1002],
        ]
    )
    true_expectation = tf.reduce_mean(a_z[:, :, None] * b_z[:, None, :], axis=0)
    computed_expectation = expectation_prod_neg(
        log_a_z=tf.math.log(a_z[:, :, None]), b_z=b_z[:, None, :]
    )
    assert computed_expectation.shape == (a_z.shape[1], b_z.shape[1])
    assert tf.experimental.numpy.allclose(true_expectation, computed_expectation)

    # check 2: with additional batch dimension
    a_z = tf.constant(
        [
            [[1.0, 2.0], [1.0, 2.0]],
            [[3.0, 4.0], [3.0, 4.0]],
            [[5.0, 6.0], [5.0, 6.0]],
            [[5.0, 6.0], [5.0, 6.0]],
            [[5.0, 6.0], [5.0, 6.0]],
        ]
    )
    b_z = tf.constant(
        [
            [[3.0, 1.0, -3.0], [3.0, 1.0, -3.0]],
            [[-2.0, 1.0, 0.0], [-2.0, 1.0, 0.0]],
            [[1.0, 2.0, -17.0], [1.0, 2.0, -17.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[-190.0, -200.0, -1002], [-190.0, -200.0, -1002]],
        ]
    )
    true_expectation = tf.reduce_mean(a_z[:, :, :, None] * b_z[:, :, None, :], axis=0)
    computed_expectation = expectation_prod_neg(
        log_a_z=tf.math.log(a_z[:, :, :, None]), b_z=b_z[:, :, None, :]
    )
    assert computed_expectation.shape == (a_z.shape[1], a_z.shape[2], b_z.shape[2])
    assert tf.experimental.numpy.allclose(true_expectation, computed_expectation)


def test_compute_S_bar():
    tf.config.run_functions_eagerly(True)

    ## check 1
    # generate relevant tensors
    S = 5
    K = 4
    D = 3
    z = tf.random.uniform(shape=(S, D))
    loc = tf.random.uniform(shape=(K, D))
    # for this test, we do not require prec to be a valid precision matrix
    prec = tf.random.uniform(shape=(K, D, D))
    log_tgt_post_grad = tf.random.uniform(shape=(S, D))

    # check computation
    S_bar = compute_S_bar(z=z, loc=loc, prec=prec, log_tgt_post_grad=log_tgt_post_grad)
    for s in range(S):
        for k in range(K):
            for i in range(D):
                for j in range(D):
                    S_bar_skij = prec[k] @ (z[s] - loc[k])[:, None]
                    S_bar_skij = S_bar_skij[i] * (-log_tgt_post_grad[s, j])
                    assert S_bar[s, k, i, j] == S_bar_skij

    ## check 2: test additional batch dimensions
    # generate relevant tensors
    S = 5
    B = 3
    C = 2
    K = 4
    D = 3
    z = tf.random.uniform(shape=(S, B, C, D))
    loc = tf.random.uniform(shape=(B, C, K, D))
    # for this test, we do not require prec to be a valid precision matrix
    prec = tf.random.uniform(shape=(B, C, K, D, D))
    log_tgt_post_grad = tf.random.uniform(shape=(S, B, C, D))

    # check computation
    S_bar = compute_S_bar(z=z, loc=loc, prec=prec, log_tgt_post_grad=log_tgt_post_grad)
    for s in range(S):
        for b in range(B):
            for c in range(C):
                for k in range(K):
                    for i in range(D):
                        for j in range(D):
                            S_bar_sbckij = (
                                prec[b, c, k] @ (z[s, b, c] - loc[b, c, k])[..., None]
                            )
                            S_bar_sbckij = S_bar_sbckij[i] * (
                                -log_tgt_post_grad[s, b, c, j]
                            )
                            assert S_bar[s, b, c, k, i, j] == S_bar_sbckij


def test_log_omega_log_w_conversion():
    tf.config.run_functions_eagerly(True)

    ### check 1
    ## check that log_w == log_omega_to_log_w(log_w_to_log_omega(log_w))
    # generate w
    w = [0.1, 0.2, 0.3, 0.2]
    w = tf.concat((w, 1.0 - tf.math.reduce_sum(w, keepdims=True)), axis=0)
    assert tf.math.reduce_sum(w) == 1.0
    # perform test
    log_w = tf.math.log(w)
    log_omega = log_w_to_log_omega(log_w)
    assert tf.experimental.numpy.allclose(
        log_omega, tf.math.log(w[:4]) - tf.math.log(w[4])
    )
    assert tf.experimental.numpy.allclose(log_w, log_omega_to_log_w(log_omega))

    ## check that log_omega == log_w_to_log_omega(log_omega_to_log_w(log_omega))
    # generate omega
    log_omega = tf.math.log(w[:4]) - tf.math.log(w[4])
    # perform test
    log_w = log_omega_to_log_w(log_omega)
    assert tf.experimental.numpy.allclose(log_omega, log_w_to_log_omega(log_w))

    ### check 2: additional batch dimensions
    ## check that log_w == log_omega_to_log_w(log_w_to_log_omega(log_w))
    # generate w
    w = [[0.1, 0.2, 0.3, 0.2], [0.1, 0.2, 0.3, 0.1]]
    w = tf.concat((w, 1.0 - tf.math.reduce_sum(w, keepdims=True, axis=-1)), axis=-1)
    assert tf.reduce_all(tf.math.reduce_sum(w, axis=-1) == 1.0)
    # perform test
    log_w = tf.math.log(w)
    log_omega = log_w_to_log_omega(log_w)
    assert tf.experimental.numpy.allclose(
        log_omega, tf.math.log(w[:, :4]) - tf.math.log(w[:, 4:])
    )
    assert tf.experimental.numpy.allclose(log_w, log_omega_to_log_w(log_omega))

    ## check that log_omega == log_w_to_log_omega(log_omega_to_log_w(log_omega))
    # generate omega
    log_omega = tf.math.log(w[:, :4]) - tf.math.log(w[:, 4:])
    # perform test
    log_w = log_omega_to_log_w(log_omega)
    assert tf.experimental.numpy.allclose(log_omega, log_w_to_log_omega(log_w))


# def test_sample_categorical():
#     n_samples = 10

#     # one component
#     log_w = tf.math.log([1.0])
#     samples = sample_categorical(n_samples=n_samples, log_w=log_w)
#     assert samples.shape == (n_samples,)
#     assert tf.reduce_all(samples == 0)

#     log_w = tf.math.log([[1.0], [1.0]])
#     samples = sample_categorical(n_samples=n_samples, log_w=log_w)
#     assert samples.shape == (n_samples, 2)
#     assert tf.reduce_all(samples == 0)

#     # two components
#     log_w = tf.math.log([0.0, 1.0])
#     samples = sample_categorical(n_samples=n_samples, log_w=log_w)
#     assert samples.shape == (n_samples,)
#     assert tf.reduce_all(samples == 1)

#     log_w = tf.math.log([1.0, 0.0])
#     samples = sample_categorical(n_samples=n_samples, log_w=log_w)
#     assert samples.shape == (n_samples,)
#     assert tf.reduce_all(samples == 0)

#     log_w = tf.math.log([[0.0, 1.0], [1.0, 0.0]])
#     samples = sample_categorical(n_samples=n_samples, log_w=log_w)
#     assert samples.shape == (n_samples, 2)
#     assert tf.reduce_all(samples[:, 0] == 1)
#     assert tf.reduce_all(samples[:, 1] == 0)

#     # three components
#     log_w = tf.math.log([0.0, 1.0, 0.0])
#     samples = sample_categorical(n_samples=n_samples, log_w=log_w)
#     assert samples.shape == (n_samples,)
#     assert tf.reduce_all(samples == 1)

#     log_w = tf.math.log([0.0, 0.0, 1.0])
#     samples = sample_categorical(n_samples=n_samples, log_w=log_w)
#     assert samples.shape == (n_samples,)
#     assert tf.reduce_all(samples == 2)

#     log_w = tf.math.log([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
#     samples = sample_categorical(n_samples=n_samples, log_w=log_w)
#     assert samples.shape == (n_samples, 2)
#     assert tf.reduce_all(samples[:, 0] == 1)
#     assert tf.reduce_all(samples[:, 1] == 0)

#     # many samples
#     n_samples = 1000000
#     log_w = tf.math.log([[0.1, 0.4, 0.5], [0.2, 0.3, 0.5]])
#     samples = sample_categorical(n_samples=n_samples, log_w=log_w)
#     assert samples.shape == (n_samples, 2)
#     for i in range(log_w.shape[0]):
#         for k in range(log_w.shape[1]):
#             cur_ratio = tf.reduce_sum(tf.cast(samples[:, i] == k, tf.int32)) / n_samples
#             assert tf.experimental.numpy.allclose(
#                 cur_ratio, tf.exp(log_w[i, k]), atol=0.01, rtol=0.0
#             )


# def test_sample_gaussian():
#     n_samples = 100000

#     # check 1: d_z == 1
#     d_z = 1
#     loc = tf.constant([1.0])
#     scale_tril = tf.constant([[0.1]])
#     samples = sample_gaussian(
#         n_samples=n_samples,
#         loc=loc,
#         scale_tril=scale_tril,
#     )
#     assert samples.shape == (n_samples, d_z)
#     empirical_mean = tf.reduce_mean(samples, axis=0)
#     empirical_std = tf.math.reduce_std(samples, axis=0)
#     assert tf.experimental.numpy.allclose(empirical_mean, loc, atol=0.01, rtol=0.0)
#     assert tf.experimental.numpy.allclose(
#         empirical_std, scale_tril, atol=0.01, rtol=0.0
#     )

#     # check 2: d_z == 2
#     d_z = 2
#     loc = tf.constant([1.0, -1.0])
#     scale_tril = tf.constant([[0.1, 0.0], [-2.0, 1.0]])
#     cov = scale_tril_to_cov(scale_tril)
#     samples = sample_gaussian(
#         n_samples=n_samples,
#         loc=loc,
#         scale_tril=scale_tril,
#     )
#     assert samples.shape == (n_samples, d_z)
#     empirical_mean = tf.reduce_mean(samples, axis=0)
#     empirical_cov = tfp.stats.covariance(samples)
#     assert tf.experimental.numpy.allclose(empirical_mean, loc, atol=0.01, rtol=0.0)
#     assert tf.experimental.numpy.allclose(empirical_cov, cov, atol=0.05, rtol=0.0)

#     # check 3: d_z == 2, batch_dim
#     d_z = 2
#     loc = tf.constant([[1.0, -1.0], [2.0, 3.0]])
#     scale_tril = tf.constant([[[0.1, 0.0], [-2.0, 1.0]], [[2.0, 0.0], [-3.0, 1.0]]])
#     cov = scale_tril_to_cov(scale_tril)
#     samples = sample_gaussian(
#         n_samples=n_samples,
#         loc=loc,
#         scale_tril=scale_tril,
#     )
#     assert samples.shape == (n_samples, 2, d_z)
#     for b in range(2):
#         cur_samples = samples[:, b, :]
#         empirical_mean = tf.reduce_mean(cur_samples, axis=0)
#         empirical_cov = tfp.stats.covariance(cur_samples)
#         assert tf.experimental.numpy.allclose(
#             empirical_mean, loc[b], atol=0.01, rtol=0.0
#         )
#         assert tf.experimental.numpy.allclose(
#             empirical_cov, cov[b], atol=0.05, rtol=0.0
#         )