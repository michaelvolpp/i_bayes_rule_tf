from i_bayes_rule.util_autograd import eval_fn_grad_hess
import tensorflow as tf


def fun_with_known_grad_hess(z: tf.Tensor):
    tf.config.run_functions_eagerly(True)

    # TODO: this function should return a different result for each dim in batch_shape
    # Implement f: R^2 -> R, f(z) = z_0**2 + 2*z_1**3 + 3*z_0*z_1 + 5
    n_samples = z.shape[0]
    batch_shape = z.shape[1:-1]
    d_z = z.shape[-1]
    assert d_z == 2

    ## compute function value
    f_z = z[..., 0] ** 2 + 2 * z[..., 1] ** 3 + 3 * z[..., 0] * z[..., 1] ** 2 + 5

    ## compute gradient
    f_z_grad = tf.stack(
        [
            # f_z_grad_0 = df/dz_0 = 2*z_0 + 3*z_1 ** 2
            2 * z[..., 0] + 3 * z[..., 1] ** 2,
            # f_z_grad_1 = df/dz_1 = 6*z_1**2 + 6*z_0*z_1
            6 * z[..., 1] ** 2 + 6 * z[..., 0] * z[..., 1],
        ],
        axis=-1,
    )

    ## compute hessian
    f_z_hess = tf.stack(
        [
            # f_z_hess_00 = d^2f/dz_0^2 = 2
            # f_z_hess_01 = d^2f/dz_0 dz_1 = 6 * z_1
            tf.stack(
                [
                    2 * tf.ones(((n_samples,) + batch_shape)),
                    6 * z[..., 1],
                ],
                axis=-1,
            ),
            # f_z_hess_10 = d^2f/dz_1 dz_0 = 6 * z_1
            # f_z_hess_11 = d^2f/dz_1^2 = 12 * z_1 + 6 * z_0
            tf.stack(
                [
                    6 * z[..., 1],
                    12 * z[..., 1] + 6 * z[..., 0],
                ],
                axis=-1,
            ),
        ],
        axis=-1,
    )

    return f_z, f_z_grad, f_z_hess


def test_eval_grad_hess():
    tf.config.run_functions_eagerly(True)

    n_samples = 10
    fun = lambda z: fun_with_known_grad_hess(z)[0]

    # check 1: no batch_dim
    z = tf.random.normal((n_samples, 2))
    f_z, f_z_grad, f_z_hess = eval_fn_grad_hess(
        fn=fun, z=z, compute_grad=True, compute_hess=True
    )
    true_f_z, true_f_z_grad, true_f_z_hess = fun_with_known_grad_hess(z)
    assert f_z.shape == (n_samples,)
    assert tf.experimental.numpy.allclose(f_z, true_f_z)
    assert f_z_grad.shape == (n_samples, 2)
    assert tf.experimental.numpy.allclose(f_z_grad, true_f_z_grad)
    assert f_z_hess.shape == (n_samples, 2, 2)
    assert tf.experimental.numpy.allclose(f_z_hess, true_f_z_hess)

    # check 2: batch_dim
    # TODO: setting this to false might make test fail for large no. of batch_dims
    tf.config.run_functions_eagerly(True)
    batch_shape = (9,)
    z = tf.random.normal((n_samples,) + batch_shape + (2,))
    f_z, f_z_grad, f_z_hess = eval_fn_grad_hess(
        fn=fun, z=z, compute_grad=True, compute_hess=True
    )
    true_f_z, true_f_z_grad, true_f_z_hess = fun_with_known_grad_hess(z)
    assert f_z.shape == (n_samples,) + batch_shape
    assert tf.experimental.numpy.allclose(f_z, true_f_z)
    assert f_z_grad.shape == (n_samples,) + batch_shape + (2,)
    assert tf.experimental.numpy.allclose(f_z_grad, true_f_z_grad)
    assert f_z_hess.shape == (n_samples,) + batch_shape + (2, 2)
    assert tf.experimental.numpy.allclose(f_z_hess, true_f_z_hess)
