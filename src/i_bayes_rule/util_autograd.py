import tensorflow as tf


def eval_fn_grad_hess(fn, z: tf.Tensor, compute_grad: bool, compute_hess: bool):
    # check input
    assert not (compute_grad == False and compute_hess == True)
    n_samples = z.shape[0]
    batch_shape = z.shape[1:-1]
    d_z = z.shape[-1]
    assert z.shape == (n_samples,) + batch_shape + (d_z,)

    if not compute_grad:
        f_z, f_z_grad, f_z_hess = _eval_fn(fn=fn, z=z), None, None
    elif compute_grad and not compute_hess:
        (f_z, f_z_grad), f_z_hess = _eval_fn_grad(fn=fn, z=z), None
    else:
        f_z, f_z_grad, f_z_hess = _eval_fn_grad_hess(fn=fn, z=z)

    # check output
    assert f_z.shape == (n_samples,) + batch_shape
    if compute_grad:
        assert f_z_grad.shape == (n_samples,) + batch_shape + (d_z,)
    if compute_hess:
        assert f_z_hess.shape == (n_samples,) + batch_shape + (d_z, d_z)

    return f_z, f_z_grad, f_z_hess


def _eval_fn(fn, z: tf.Tensor):
    # compute fn(z)
    f_z = fn(z)
    return f_z


def _eval_fn_grad(fn, z: tf.Tensor):
    # compute fn(z), and its gradient
    with tf.GradientTape(watch_accessed_variables=False) as g2:
        g2.watch(z)
        f_z = fn(z)
    f_z_grad = g2.gradient(f_z, z)

    return f_z, f_z_grad


def _eval_fn_grad_hess(fn, z: tf.Tensor):
    # TODO: fully batch the Hessian computation
    #  The problem is that batch_jacobian allows only one batch dimension. If
    #  multiple batch dimensions (e.g., sample-dim and task-dim) are present,
    #  batch jacobian does unneccesary computations akin to the ones described
    #  here: https://www.tensorflow.org/guide/advanced_autodiff#batch_jacobian
    #  Flattening the batch-dims **after** gradient computation and before calling
    #  batch jacobian is not an option as tf creates new variables for reshaped
    #  variables, losing track of gradients.
    #  Flattening the batch-dims **before** all computations is not an option
    #  because fn(z) has to be called with correct batch dims, as in general
    #  it performs different computations for each batch dim.

    # compute fn(z), and its gradient and hessian

    # check input
    n_samples = z.shape[0]
    batch_shape = z.shape[1:-1]
    d_z = z.shape[-1]
    assert not len(batch_shape) > 1  # not implemented yet

    # # flatten batch dimensions
    # z = tf.reshape(z, (-1, d_z))

    with tf.GradientTape(watch_accessed_variables=False, persistent=True) as g1:
        g1.watch(z)
        with tf.GradientTape(watch_accessed_variables=False) as g2:
            g2.watch(z)
            f_z = fn(z)
        f_z_grad = g2.gradient(f_z, z)
    f_z_hess = g1.batch_jacobian(f_z_grad, z)  # TODO: performs unneccesary computations
    if len(batch_shape) == 1:
        f_z_hess = tf.einsum("sbxby->sbxy", f_z_hess)

    # unflatten batch dimensions
    # f_z = tf.reshape(f_z, (n_samples,) + batch_shape)
    # f_z_grad = tf.reshape(f_z_grad, (n_samples,) + batch_shape + (d_z,))
    # f_z_hess = tf.reshape(f_z_hess, (n_samples,) + batch_shape + (d_z, d_z))

    return f_z, f_z_grad, f_z_hess
