import torch


def exp_reducer(x):
    return x.exp().sum(dim=-1)  # f: R^n -> R^(n-1)


torch.manual_seed(123)
inputs = torch.rand(2, 2)
jac = torch.autograd.functional.jacobian(exp_reducer, inputs, vectorize=True)

print(inputs)
print(exp_reducer(inputs))
print(inputs.exp())
print(jac.shape)
print(jac)
