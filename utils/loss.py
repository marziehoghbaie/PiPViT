import torch
import torch.nn.functional as F

"""
The contrastive loss functions are borrowed from PiP-Net repo.
"""
def calculate_loss(proto_features, pooled, align_pf_weight, t_weight, unif_weight, EPS=1e-10, tanh_type='avg'):

    pooled1, pooled2 = pooled.chunk(2)
    pf1, pf2 = proto_features.chunk(2)
    embv2 = pf2.flatten(start_dim=2).permute(0, 2, 1).flatten(end_dim=1)
    embv1 = pf1.flatten(start_dim=2).permute(0, 2, 1).flatten(end_dim=1)

    a_loss_pf = (align_loss(embv1, embv2.detach()) + align_loss(embv2, embv1.detach())) / 2.
    tanh_loss = -(torch.log(torch.tanh(torch.sum(pooled1, dim=0)) + EPS).mean() + torch.log(
    torch.tanh(torch.sum(pooled2, dim=0)) + EPS).mean()) / 2.

    loss = align_pf_weight * a_loss_pf
    loss += t_weight * tanh_loss
    uni_loss = (uniform_loss(F.normalize(pooled1 + EPS, dim=1)) + uniform_loss(
        F.normalize(pooled2 + EPS, dim=1))) / 2.
    loss += unif_weight * uni_loss

    return loss


# Extra uniform loss from https://www.tongzhouwang.info/hypersphere/.
def uniform_loss(x, t=2):
    loss = (torch.pdist(x, p=2).pow(2).mul(-t).exp().mean() + 1e-10).log()
    return loss


# from https://gitlab.com/mipl/carl/-/blob/main/losses.py
def align_loss(inputs, targets, EPS=1e-12):
    assert inputs.shape == targets.shape
    assert targets.requires_grad == False
    loss = torch.einsum("nc,nc->n", [inputs, targets])
    loss = -torch.log(loss + EPS).mean()
    return loss


def entropy_loss(p, EPS=1e-10):
    return torch.distributions.Categorical(probs=p).entropy().mean()