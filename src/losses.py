"""
Loss functions (from scratch).

Total loss for CAGCo-Net:
    L = BCE_with_pos_weight(logit, y) + lambda_ccl * L_ccl(z, y, pairs)

L_ccl is a confounder-weighted supervised contrastive loss. It is SupCon
(Khosla et al., 2020) with one change: in the denominator, the anchor's
recovered confounder partner (same image + different caption, or same caption
+ different image, opposite label) is up-weighted by beta. These are exactly
the "hard negatives" that fool unimodal shortcuts, so the model is pushed to
separate memes that differ by a single modality.

For anchor i with positives P(i) (same label, j != i):
    L_i = -1/|P(i)| * sum_{p in P(i)} log( exp(s_ip) / sum_{j != i} w_ij exp(s_ij) )
    s_ij = z_i . z_j / tau,  w_ij = beta if (i,j) is a confounder pair else 1
"""
import torch
import torch.nn.functional as F


def bce_loss(logit, y, pos_weight):
    return F.binary_cross_entropy_with_logits(logit, y.float(), pos_weight=pos_weight)


def confounder_supcon(z, y, pair_mask, tau: float = 0.1, beta: float = 3.0):
    """z: (B,p) L2-normalised, y: (B,), pair_mask: (B,B) bool."""
    B = z.size(0)
    eye = torch.eye(B, dtype=torch.bool, device=z.device)
    sim = (z @ z.t()) / tau
    sim = sim.masked_fill(eye, -1e9)                                    # drop self-similarity
    log_w = torch.zeros_like(sim).masked_fill(pair_mask, float(torch.log(torch.tensor(beta))))
    log_denom = torch.logsumexp(sim + log_w, dim=1, keepdim=True)       # stable log-sum-exp
    log_prob = sim - log_denom
    pos = (y[:, None] == y[None, :]) & ~eye
    n_pos = pos.sum(1)
    valid = n_pos > 0
    if valid.sum() == 0:
        return z.new_zeros(())
    loss_i = -(log_prob * pos).sum(1) / n_pos.clamp(min=1)
    return loss_i[valid].mean()
