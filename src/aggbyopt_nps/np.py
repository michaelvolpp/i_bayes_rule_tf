import torch

from aggbyopt_nps.util import (
    covs_to_precs,
    precs_to_covs,
    eval_fn_grad_hess,
    expectation_prod_neg,
    precs_to_prec_trils,
    log_omegas_to_log_weights,
    log_weights_to_log_omegas,
)


class NP(torch.nn.module):
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
        self.gmm_weights, self.gmm_means, self.gmm_covs = None, None, None

    def reset_gmm(self, n_tasks: int):
        self.n_tasks = n_tasks
        (
            self.gmm_weights,
            self.gmm_means,
            self.gmm_covs,
        ) = create_initial_gmm_parameters(
            n_tasks=n_tasks,
            d_z=self.d_z,
            n_components=self.gmm_n_components,
            prior_scale=self.gmm_prior_scale,
        )

    def log_likelihood(self, x_t: torch.tensor, y_t: torch.tensor, z: torch.tensor):
        """
        log p(D^t | z, theta)
        """
        # check input
        n_points = x_t.shape[1]
        assert x_t.shape == (self.n_tasks, n_points, self.d_x)
        assert y_t.shape == (self.n_tasks, n_points, self.d_y)
        assert z.shape == (self.n_tasks, self.d_z)

        # compute log likelihood
        xz = torch.stack(
            (x_t, z[:, None, :].expand(-1, n_points, -1)),
            dim=2,
        )
        assert xz.shape == (self.n_tasks, n_points, self.d_x + self.d_z)
        mu = self.decoder(xz)
        assert mu.shape == (self.n_tasks, n_points, self.d_y)
        gaussian = torch.distributions.Independent(
            torch.distributions.Normal(loc=mu, scale=self.decoder_output_scale),
            reinterpreted_batch_ndims=1,
        )
        log_likelihood = gaussian.log_prob(y_t)

        # check output
        assert log_likelihood.shape == (self.n_tasks, n_points)
        return log_likelihood

    def log_conditional_prior_density(self, z: torch.tensor):
        """
        log p(z | D^c, theta)
        """
        # check input
        assert z.shape == (self.n_tasks, self.d_z)

        # compute log conditional prior density
        gmm = create_gmm(
            weights=self.gmm_weights,
            means=self.gmm_means,
            covs=self.gmm_covs,
        )
        log_density = gmm.log_prob(z)

        # check output
        assert log_density.shape == (self.n_tasks)
        return log_density

    def log_density(self, x_t: torch.tensor, y_t: torch.tensor, z: torch.tensor):
        """
        log p(D^t | z, theta) + log p(z | D^c, theta)
        """
        return (
            self.log_likelihood(x_t=x_t, y_t=y_t, z=z)
            + self.log_conditional_prior_density(z=z),
        )

    def sample_z(self, n_samples: int):
        gmm = create_gmm(
            weights=self.gmm_weights,
            means=self.gmm_means,
            covs=self.gmm_covs,
        )
        z = gmm.sample_n(n_samples)

        # check output
        assert z.shape == (n_samples, self.d_z)
        return z


class ConditionalPriorLearner:
    def __init__(self, model, lr, n_samples):
        self.model = model
        self.lr = lr
        self.n_samples = n_samples

    def step(self, x_t, y_t, x_c, y_c):
        ## check input
        n_tasks = x_t.shape[0]
        n_points = x_t.shape[1]
        d_x = x_t.shape[2]
        d_y = y_t.shape[2]
        assert x_c.shape[0] == y_c.shape[0] == y_t.shape[0] == n_tasks
        assert x_c.shape[0] == y_c.shape[0] == y_t.shape[1] == n_points
        assert x_c.shape[2] == d_x
        assert y_c.shape[2] == d_y
        assert d_x == self.model.d_x
        assert d_y == self.model.d_y

        ## step
        z = self.model.sample_z(n=self.n_samples)
        assert z.shape == (self.n_samples, n_tasks, self.model.d_z)
        (self.model.mu, self.model.cov, self.model.log_w) = gmm_learner_step(
            log_w=self.model.log_w,
            means=self.model.mu,
            covs=self.model.cov,
            z=z,
            log_target_density_fn=self.model.log_density,
            log_model_density_fn=self.model.log_conditional_prior_density,
            log_model_component_densities_fn=self.model.log_conditional_prior_component_densities,
        )


class LikelihoodLearner:
    def __init__(self, model, lr, n_samples):
        self.model = model
        self.lr = lr
        self.n_samples = n_samples
        self.optim = torch.optim.Adam(params=self.model.theta, lr=self.lr)

    def step(self, x_t, y_t):
        ## check input
        n_tasks = x_t.shape[0]
        n_points = x_t.shape[1]
        d_x = x_t.shape[2]
        d_y = y_t.shape[2]
        assert y_t.shape[0] == n_tasks
        assert y_t.shape[1] == n_points
        assert d_x == self.model.d_x
        assert d_y == self.model.d_y

        ## perform step
        self.optim.zero_grad()
        # sample model
        z = self.model.sample_z(n=self.n_samples)
        assert z.shape == (self.n_samples, n_tasks, self.model.d_z)
        # compute likelihood
        ll = self.model.log_likelihood(x_t=x_t, y_t=y_t, z=z)
        assert ll.shape == (self.n_samples, n_tasks)
        # compute loss
        loss = -torch.log(self.n_samples)
        loss = loss + torch.logsumexp(ll, dim=0, keepdim=True)
        loss = torch.sum(loss, dim=1, keepdim=True)
        assert loss.shape == (1, 1)
        loss.squeeze()
        # step optimizer
        loss.backward()
        self.optim.step()

        return loss


def create_mlp(
    d_x: int,
    d_y: int,
    n_hidden: int,
    d_hidden: int,
) -> torch.nn.Module:
    """
    Generate a standard MLP.
    """

    layers = []
    if n_hidden == 0:  # linear model
        layers.append(torch.nn.Linear(in_features=d_x, out_features=d_y))
    else:  # fully connected MLP
        layers.append(torch.nn.Linear(in_features=d_x, out_features=d_hidden))
        layers.append(torch.nn.ReLU())
        for _ in range(n_hidden - 1):
            layers.append(torch.nn.Linear(in_features=d_hidden, out_features=d_hidden))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(in_features=d_hidden, out_features=d_y))
    net = torch.nn.Sequential(*layers)

    return net


def create_initial_gmm_parameters(d_z, n_tasks, n_components, prior_scale):
    # TODO: check more sophisticated initializations
    prior = torch.distributions.Normal(
        loc=torch.zeros(d_z), scale=prior_scale * torch.ones(d_z)
    )
    initial_cov = prior_scale ** 2 * torch.eye(d_z)  # same as prior covariance

    weights = torch.ones((n_tasks, n_components)) / n_components
    means = prior.sample((n_tasks, n_components))
    covs = initial_cov.repeat((n_tasks, n_components, 1, 1))

    # check output
    assert weights.shape == (n_tasks, n_components)
    assert means.shape == (n_tasks, n_components, d_z)
    assert covs.shape == (n_tasks, n_components, d_z, d_z)
    return weights, means, covs


def create_gmm(weights, means, covs):
    # TODO: speed this up
    mix = torch.distributions.Categorical(weights)
    comp = torch.distributions.MultivariateNormal(loc=means, covariance_matrix=covs)
    gmm = torch.distributions.MixtureSameFamily(mix, comp)

    return gmm


def gmm_learner_step(
    z: torch.tensor,
    weights: torch.tensor,
    means: torch.tensor,
    covs: torch.tensor,
    log_target_density_fn,
    log_model_density_fn,
    log_model_component_densities_fn,
    lr_w: float,
    lr_mean_prec: float,
):
    ## check input
    n_tasks = weights.shape[0]
    n_components = weights.shape[1]
    n_samples = z.shape[1]
    d_z = means.shape[2]
    assert z.shape == (n_samples, n_tasks, d_z)
    assert weights.shape == (n_tasks, n_components)
    assert means.shape == (n_tasks, n_components, d_z)
    assert covs.shape == (n_tasks, n_components, d_z, d_z)

    ## transform inputs
    log_weights = torch.log(weights)
    # TODO: stay in one domain!
    precs = covs_to_precs(covs)
    prec_trils = precs_to_prec_trils(precs)
    assert log_weights.shape == (n_tasks, n_components)
    assert precs.shape == (n_tasks, n_components, d_z, d_z)
    assert prec_trils.shape == (n_tasks, n_components, d_z, d_z)

    ## evaluate target distribution
    (
        log_target_density,
        log_target_density_grad,
        log_target_density_hess,
    ) = eval_fn_grad_hess(
        fn=log_target_density_fn, z=z, compute_grad=True, compute_hess=True
    )
    assert log_target_density.shape == (n_samples, n_tasks)
    assert log_target_density_grad.shape == (n_samples, n_tasks, d_z)
    assert log_target_density_hess.shape == (n_samples, n_tasks, d_z, d_z)

    ## eval model distribution
    (
        log_model_density,
        log_model_density_grad,
        log_model_density_hess,
    ) = eval_fn_grad_hess(
        fn=log_model_density_fn, z=z, compute_grad=True, compute_hess=True
    )
    assert log_model_density.shape == (n_samples, n_tasks)
    assert log_model_density_grad.shape == (n_samples, n_tasks, d_z)
    assert log_model_density_hess.shape == (n_samples, n_tasks, d_z, d_z)

    ## compute delta(z)
    # TODO: Lin et al. do this somehow differently
    log_delta_z = log_model_component_densities_fn(z=z) - log_model_density[:, :, None]
    assert log_delta_z.shape == (n_samples, n_tasks, n_components)

    # TODO: learning rate annealing?

    ## update model parameters
    new_log_weights = update_log_weights(
        log_weights=log_weights,
        log_delta_z=log_delta_z,
        log_target_density=log_target_density,
        log_model_density=log_model_density,
        lr=lr_w,
    )
    new_weights = torch.exp(new_log_weights)
    new_means = update_means(
        means=means,
        prec_trils=prec_trils,
        log_delta_z=log_delta_z,
        log_target_density_grad=log_target_density_grad,
        log_model_density_grad=log_model_density_grad,
        lr=lr_mean_prec,
    )
    # TODO: first order version
    g = compute_g_hessian(
        log_delta_z=log_delta_z,
        log_target_density_hess=log_target_density_hess,
        log_model_density_hess=log_model_density_hess,
    )
    assert g.shape == (n_tasks, n_components, d_z, d_z)
    new_precs = update_precs(
        precs=precs,
        prec_trils=prec_trils,
        g=g,
        lr=lr_mean_prec,
    )
    new_covs = precs_to_covs(new_precs)

    # check output
    assert new_weights.shape == (n_tasks, n_components)
    assert new_means.shape == (n_tasks, n_components, d_z)
    assert new_covs.shape == (n_tasks, n_components, d_z, d_z)

    return new_weights, new_means, new_covs


def update_log_weights(
    log_weights: torch.tensor,
    log_delta_z: torch.tensor,
    log_target_density: torch.tensor,
    log_model_density: torch.tensor,
    lr: float,
):
    # check inputs
    n_tasks = log_weights.shape[0]
    n_components = log_weights.shape[1]
    n_samples = log_target_density.shape[1]
    assert log_weights.shape == (n_tasks, n_components)
    assert log_delta_z.shape == (n_samples, n_tasks, n_components)
    assert log_target_density.shape == (n_samples, n_tasks)
    assert log_model_density.shape == (n_samples, n_tasks)

    # computations are performed in log_omega space
    #  where omega[k] := w[k] / w[K] for k=1..K-1
    log_omegas = log_weights_to_log_omegas(log_weights)
    assert log_omegas.shape == (n_tasks, n_components)

    ## update log_omega
    b_z = -log_target_density + log_model_density
    # compute E_q[delta(z) * b(z)]
    expectation = expectation_prod_neg(log_a_z=log_delta_z, b_z=b_z[:, :, None])
    assert expectation.shape == (n_tasks, n_components)
    # compute E_q[(delta(z)[:-1] - delta(z)[-1])*b(z)]
    d_log_omegas = expectation[:, :-1] - expectation[:, -1]
    # update log_omega
    d_log_omegas = -lr * d_log_omegas
    log_omegas = log_omegas + d_log_omegas

    # go back to log_w space
    log_weights = log_omegas_to_log_weights(log_omegas)

    # check outputs
    assert log_weights.shape == (n_tasks, n_components)
    return log_weights


def update_means(
    means: torch.tensor,
    prec_trils: torch.tensor,
    log_delta_z: torch.tensor,
    log_target_density_grad: torch.tensor,
    log_model_density_grad: torch.tensor,
    lr: float,
):
    # check inputs
    n_tasks = means.shape[0]
    n_components = means.shape[1]
    n_samples = log_target_density_grad.shape[1]
    d_z = means.shape[2]
    assert means.shape == (n_tasks, n_components, d_z)
    assert prec_trils.shape == (n_tasks, n_components, d_z, d_z)
    assert log_delta_z.shape == (n_samples, n_tasks, n_components)
    assert log_target_density_grad.shape == (n_samples, n_tasks, d_z)
    assert log_model_density_grad.shape == (n_samples, n_tasks, d_z)

    ## update mu
    b_z_grad = -log_target_density_grad + log_model_density_grad
    # compute E_q[delta(z) * d/dz(b(z))]
    expectation = expectation_prod_neg(
        log_a_z=log_delta_z[:, :, :, None], b_z=b_z_grad[:, :, None, :]
    )
    assert expectation.shape == (n_tasks, n_components, d_z)
    # compute S^{-1} * E_q[delta(z)*grad_z(b(z))]
    d_means = torch.linalg.cholesky_solve(
        chol=prec_trils,
        rhs=expectation[:, :, :, None],
    )[:, :, 0]
    assert d_means.shape == (n_tasks, n_components, d_z)
    # update mu
    d_means = -lr * d_means
    means = means + d_means

    # check outputs
    assert means.shape == (n_tasks, n_components, d_z)
    return means


def compute_g_hessian(
    log_delta_z,
    log_target_density_hess,
    log_model_density_hess,
):
    # check inputs
    n_tasks = log_delta_z.shape[0]
    n_samples = log_delta_z.shape[1]
    n_components = log_delta_z.shape[2]
    d_z = log_target_density_hess.shape[2]
    assert log_delta_z.shape == (n_samples, n_tasks, n_components)
    assert log_target_density_hess.shape == (n_samples, n_tasks, d_z, d_z)
    assert log_model_density_hess.shape == (n_samples, n_tasks, d_z, d_z)

    # compute g = -E_q[delta(z) * d^2/dz^2 b(z)]
    b_z_hess = -log_target_density_hess + log_model_density_hess
    g = -expectation_prod_neg(
        log_a_z=log_delta_z[:, :, :, None, None], b_z=b_z_hess[:, :, None, :, :]
    )

    # check output
    assert g.shape == (n_tasks, n_components, d_z, d_z)
    return g


def update_precs(
    precs: torch.tensor,
    prec_trils: torch.tensor,
    g: torch.tensor,
    lr: float,
):
    # check input
    n_tasks = precs.shape[0]
    n_components = precs.shape[1]
    d_z = precs.shape[2]
    assert precs.shape == (n_tasks, n_components, d_z, d_z)
    assert prec_trils.shape == (n_tasks, n_components, d_z, d_z)
    assert g.shape == (n_tasks, n_components, d_z, d_z)

    ## update precision
    # solve linear equations
    sols = torch.linalg.cholesky_solve(chol=prec_trils, rhs=g)
    assert sols.shape == (n_tasks, n_components, d_z, d_z)
    # compute update
    d_precs = -lr * g
    d_precs = d_precs + 0.5 * lr ** 2 * torch.einsum("kij,kjl->kil", g, sols)
    assert d_precs.shape == (n_tasks, n_components, d_z, d_z)
    precs = precs + d_precs

    ## TODO: this might be more stable?
    # U = tf.transpose(prec_tril, [0, 2, 1]) - lr * tf.linalg.cholesky_solve(
    #     chol=prec_tril, rhs=g
    # )
    # prec = 0.5 * (prec + tf.transpose(U, [0, 2, 1]) @ U)
    # prec = 0.5 * (prec + tf.transpose(prec, [0, 2, 1]))

    # check output
    assert precs.shape == (n_tasks, n_components, d_z, d_z)
    return precs


def meta_train_np(meta_dataloader, np_model, n_iter, callback):
    cp_learner = ConditionalPriorLearner(model=np_model)
    th_learner = LikelihoodLearner(model=np_model)
    for iter in range(n_iter):
        for batch in meta_dataloader:
            cp_learner.step(x_t=batch.x_t, y_t=batch.y_t, x_c=batch.x_c, y_c=batch.y_c)
            th_learner.step(x_t=batch.x_t, y_t=batch.y_t, x_c=batch.x_c, y_c=batch.y_c)

        callback(iter=iter, np_model=np_model)
