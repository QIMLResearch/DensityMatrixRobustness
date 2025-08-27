import torch
import torch.nn.functional as F
from defenses.constraints import enforce_dm_conditions, enforce_stats_conditions


def pgd_attack(model,
               data, 
               labels, 
               device,
               encoding=None,
               constrain=False,
               eps=0.3, 
               alpha=0.01, 
               iters=40
               ):

    perturbed_data = data.clone().detach().to(device)
    perturbed_data.requires_grad = True
    
    model.train()

    for _ in range(iters):
        outputs = model(perturbed_data)
        loss = F.cross_entropy(outputs, labels)

        model.zero_grad()
        loss.backward()

        perturbed_data = perturbed_data + alpha * perturbed_data.grad.sign()
        
        def clamp_perturbation(perturbed_data, data, eps):
            """
            Helper function to clamp perturbation within [-epsilon, epsilon].
            """
            perturbation = perturbed_data - data
            perturbed_data = torch.clamp(perturbed_data, -eps, eps)
            return data + perturbation

        if encoding in ['DM', 'Stats']:
            if constrain:
                constraint_fn = enforce_dm_conditions if encoding == 'DM' else enforce_stats_conditions
                perturbed_data = constraint_fn(perturbed_data)
                perturbed_data = clamp_perturbation(perturbed_data, data, eps)
            else:
                perturbed_data = torch.clamp(perturbed_data, data - eps, data + eps)
        else:
            perturbed_data = torch.clamp(perturbed_data, data - eps, data + eps)

        perturbed_data = perturbed_data.detach()
        perturbed_data.requires_grad = True

    return perturbed_data