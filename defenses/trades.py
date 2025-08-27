import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from defenses.constraints import enforce_dm_conditions, enforce_stats_conditions


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                encoding=None,
                constrain=False,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    """
    TRADES Loss: Balances natural loss and robust loss for adversarial training.
    """

    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    model.eval()
    batch_size = len(x_natural)
    

    x_adv = x_natural.detach() + 0.001 * torch.randn_like(x_natural).to(x_natural.device).detach()
    
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
                
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * grad.sign()
            
            def clamp_perturbation(x_adv, x_natural, epsilon):
                """
                Helper function to clamp perturbation within [-epsilon, epsilon].
                """
                perturbation = x_adv - x_natural
                perturbation = torch.clamp(perturbation, -epsilon, epsilon)
                return x_natural + perturbation

            if encoding in ['DM', 'Stats']:
                if constrain:
                    constraint_fn = enforce_dm_conditions if encoding == 'DM' else enforce_stats_conditions
                    x_adv = constraint_fn(x_adv)
                    x_adv = clamp_perturbation(x_adv, x_natural, epsilon)
                else:
                    x_adv = torch.clamp(x_adv, x_natural - epsilon, x_natural + epsilon)
            else:
                x_adv = torch.clamp(x_adv, x_natural - epsilon, x_natural + epsilon)

            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    model.train()
    x_adv = Variable(x_adv, requires_grad=False)
    optimizer.zero_grad()
    
    logits_nat = model(x_natural)
    logits_adv = model(x_adv)
    
    loss_natural = F.cross_entropy(logits_nat, y)

    loss_robust = criterion_kl(F.log_softmax(logits_adv, dim=1),
                               F.softmax(logits_nat, dim=1))
    
    loss = loss_natural + beta * loss_robust
    return loss

    
