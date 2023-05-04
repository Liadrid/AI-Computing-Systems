import torch


def fsgm_attack(data, epsilon, grad):
    sign_grad = grad.sign()
    perturbed_data = data + epsilon * sign_grad
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    return perturbed_data


def fgm_attack(data, epsilon, grad):
    grad_2 = torch.norm(grad, p=2)
    perturbed_data = data + epsilon * (grad / grad_2)
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    return perturbed_data
