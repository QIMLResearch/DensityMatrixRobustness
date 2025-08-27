import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from defenses.constraints import enforce_dm_conditions, enforce_stats_conditions


def mart_loss(model,
              x_natural,
              y,
              optimizer,
              encoding=None,
              constrain=False,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              beta=6.0,
              distance='l_inf'):
    
    
    kl = nn.KLDivLoss(reduction='none')
    model.eval()
    batch_size = len(x_natural)

    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape, device=x_natural.device).detach()

    if distance == 'l_inf':
        for step in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
                
                
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            grad = torch.clamp(grad, min=-1.0, max=1.0) 
            x_adv = x_adv.detach() + step_size * torch.sign(grad)
            

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
                    x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)       
                     
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
        
    model.train()

    x_adv = Variable(x_adv, requires_grad=False)
    optimizer.zero_grad()

    logits = model(x_natural)
    logits_adv = model(x_adv)
    
    logits_adv = torch.clamp(logits_adv, min=-50.0, max=50.0)
    logits = torch.clamp(logits, min=-50.0, max=50.0)

    adv_probs = F.softmax(logits_adv, dim=1)
    nat_probs = F.softmax(logits, dim=1)
    
    adv_probs = torch.clamp(adv_probs, min=1e-12, max=1.0)
    nat_probs = torch.clamp(nat_probs, min=1e-12, max=1.0)
    

    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

    loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

    true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
    loss = loss_adv + float(beta) * loss_robust

    return loss

