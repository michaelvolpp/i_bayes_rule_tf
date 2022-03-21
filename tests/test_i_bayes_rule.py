import tensorflow as tf
from i_bayes_rule.util import (
    compute_S_bar,
    expectation_prod_neg,
    log_omega_to_log_w,
    log_w_to_log_omega,
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
