"""
Models (Criteria 2, 4 and 5).

Everything in this file is written from scratch with basic PyTorch tensor ops
and nn.Linear / nn.LayerNorm only - no nn.MultiheadAttention, no
nn.Transformer, no pretrained heads. The frozen CLIP features come from
extract_features.py.

Proposed model
  CAGCoNet           Confounder-Aware Gated Co-Attention Network

CLIP-feature benchmarks (same frozen CLIP features as CAGCo-Net)
  EarlyFusionMLP     concatenate pooled embeddings -> MLP
  LateFusion         separate image / text heads, average their logits
  HateCLIPper        outer-product feature-interaction matrix (Kumar & Nandakumar, 2022)

Fine-tuned benchmarks (image-only ResNet, text-only BERT, VisualBERT) live in
finetune_baselines.py because they train their own backbones end to end.
ImageOnlyMLP / TextOnlyMLP are kept as optional linear-probe extras.

Ablations
  CAGCoNet(use_coattn=False)   gate + product fusion on pooled embeddings only
  CAGCoNet trained with lambda_ccl = 0  (set in train.py)

Every model takes a batch dict with keys
  img_tok (B,50,512)  txt_tok (B,L,512)  txt_mask (B,L)  img (B,512)  txt (B,512)
and returns a dict with at least "logit" (B,).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# Building blocks (from scratch)
# =========================================================================
class MultiHeadCrossAttention(nn.Module):
    """Scaled dot-product attention with h heads, written out explicitly.

    queries x_q (B,N,d) attend over keys/values x_kv (B,M,d).
    kv_mask (B,M) is True for real tokens; padded keys get -inf before softmax.
    Cost: O(B * h * N * M * d_k) time and O(B * h * N * M) memory for the map.
    """

    def __init__(self, d: int, h: int, dropout: float):
        super().__init__()
        assert d % h == 0, "d_model must be divisible by n_heads"
        self.h, self.dk = h, d // h
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(d, d) for _ in range(4))
        self.drop = nn.Dropout(dropout)

    def forward(self, x_q, x_kv, kv_mask=None):
        B, N, d = x_q.shape
        M = x_kv.shape[1]
        q = self.wq(x_q).view(B, N, self.h, self.dk).transpose(1, 2)    # (B,h,N,dk)
        k = self.wk(x_kv).view(B, M, self.h, self.dk).transpose(1, 2)   # (B,h,M,dk)
        v = self.wv(x_kv).view(B, M, self.h, self.dk).transpose(1, 2)   # (B,h,M,dk)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dk)           # (B,h,N,M)
        if kv_mask is not None:
            scores = scores.masked_fill(~kv_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attn = scores.softmax(dim=-1)
        out = (self.drop(attn) @ v).transpose(1, 2).reshape(B, N, d)    # (B,N,d)
        return self.wo(out), attn


class FeedForward(nn.Module):
    def __init__(self, d: int, dropout: float, mult: int = 2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, mult * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(mult * d, d))

    def forward(self, x):
        return self.net(x)


class CoAttentionBlock(nn.Module):
    """One bidirectional co-attention layer.

    Text tokens query image patches and image patches query text tokens in
    parallel (both use the inputs of this layer), then each stream gets a
    residual + LayerNorm and a feed-forward sub-layer (post-LN transformer style).
    """

    def __init__(self, d: int, h: int, dropout: float):
        super().__init__()
        self.t2v = MultiHeadCrossAttention(d, h, dropout)   # text attends to image
        self.v2t = MultiHeadCrossAttention(d, h, dropout)   # image attends to text
        self.ln_t1, self.ln_v1, self.ln_t2, self.ln_v2 = (nn.LayerNorm(d) for _ in range(4))
        self.ff_t, self.ff_v = FeedForward(d, dropout), FeedForward(d, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, t, v, t_mask):
        t_att, a_t2v = self.t2v(t, v)                    # every image token is valid
        v_att, a_v2t = self.v2t(v, t, kv_mask=t_mask)    # ignore padded caption tokens
        t = self.ln_t1(t + self.drop(t_att))
        v = self.ln_v1(v + self.drop(v_att))
        t = self.ln_t2(t + self.drop(self.ff_t(t)))
        v = self.ln_v2(v + self.drop(self.ff_v(v)))
        return t, v, a_t2v, a_v2t


class AttentionPool(nn.Module):
    """Additive attention pooling: a_i = softmax(w^T tanh(W x_i)), out = sum a_i x_i."""

    def __init__(self, d: int):
        super().__init__()
        self.proj = nn.Linear(d, d)
        self.score = nn.Linear(d, 1, bias=False)

    def forward(self, x, mask=None):
        s = self.score(torch.tanh(self.proj(x))).squeeze(-1)          # (B,N)
        if mask is not None:
            s = s.masked_fill(~mask, torch.finfo(s.dtype).min)
        a = s.softmax(-1)
        return (a.unsqueeze(-1) * x).sum(1), a


def mlp_head(d_in: int, d_hidden: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d_hidden), nn.GELU(),
                         nn.Dropout(dropout), nn.Linear(d_hidden, 1))


# =========================================================================
# Proposed model
# =========================================================================
class CAGCoNet(nn.Module):
    """Confounder-Aware Gated Co-Attention Network.

    1. Project CLIP image tokens and caption tokens into a shared d-dim space
       and add a learned modality-type embedding.
    2. L bidirectional co-attention blocks let each caption word look at image
       patches and each patch look at caption words.
    3. Attention-pool each stream and add a residual from the pooled CLIP
       embedding (keeps the strong global CLIP signal).
    4. A learned gate g = sigmoid(W [h_img ; h_txt] + b) re-weights the two
       streams per meme; fusion = [g*h_img ; (1-g)*h_txt ; h_img * h_txt].
    5. MLP classifier -> hateful logit.  A projection head gives z for the
       confounder-weighted contrastive loss (losses.py).
    """

    def __init__(self, in_dim=512, d=256, heads=4, layers=1, dropout=0.2, proj_dim=128,
                 use_coattn=True):
        super().__init__()
        self.use_coattn = use_coattn
        self.g_img = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d))
        self.g_txt = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d))
        if use_coattn:
            self.in_img = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d), nn.Dropout(dropout))
            self.in_txt = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d), nn.Dropout(dropout))
            self.type_emb = nn.Parameter(torch.zeros(2, d))
            nn.init.normal_(self.type_emb, std=0.02)
            self.blocks = nn.ModuleList(CoAttentionBlock(d, heads, dropout) for _ in range(layers))
            self.pool_img, self.pool_txt = AttentionPool(d), AttentionPool(d)
        self.gate = nn.Linear(2 * d, 1)
        self.classifier_body = nn.Sequential(nn.LayerNorm(3 * d), nn.Linear(3 * d, d), nn.GELU(),
                                             nn.Dropout(dropout))
        self.classifier_out = nn.Linear(d, 1)
        self.proj = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, proj_dim))

    def forward(self, b):
        h_img, h_txt = self.g_img(b["img"]), self.g_txt(b["txt"])
        out = {}
        if self.use_coattn:
            v = self.in_img(b["img_tok"]) + self.type_emb[0]
            t = self.in_txt(b["txt_tok"]) + self.type_emb[1]
            for blk in self.blocks:
                t, v, a_t2v, a_v2t = blk(t, v, b["txt_mask"])
            pv, _ = self.pool_img(v)
            pt, _ = self.pool_txt(t, b["txt_mask"])
            h_img, h_txt = h_img + pv, h_txt + pt
            out["attn_t2v"] = a_t2v                          # (B,h,L,50) for visualisation
        g = torch.sigmoid(self.gate(torch.cat([h_img, h_txt], -1)))   # (B,1)
        fused = torch.cat([g * h_img, (1 - g) * h_txt, h_img * h_txt], -1)
        hidden = self.classifier_body(fused)
        out["logit"] = self.classifier_out(hidden).squeeze(-1)
        out["z"] = F.normalize(self.proj(hidden), dim=-1)
        out["gate"] = g.squeeze(-1)
        return out


# =========================================================================
# Benchmarks
# =========================================================================
class ImageOnlyMLP(nn.Module):
    def __init__(self, in_dim=512, d=256, dropout=0.2, **_):
        super().__init__()
        self.head = mlp_head(in_dim, d, dropout)

    def forward(self, b):
        return {"logit": self.head(b["img"]).squeeze(-1)}


class TextOnlyMLP(nn.Module):
    def __init__(self, in_dim=512, d=256, dropout=0.2, **_):
        super().__init__()
        self.head = mlp_head(in_dim, d, dropout)

    def forward(self, b):
        return {"logit": self.head(b["txt"]).squeeze(-1)}


class EarlyFusionMLP(nn.Module):
    """Concat-style baseline: [img ; txt] -> MLP. No cross-modal interaction term."""

    def __init__(self, in_dim=512, d=256, dropout=0.2, **_):
        super().__init__()
        self.head = mlp_head(2 * in_dim, d, dropout)

    def forward(self, b):
        return {"logit": self.head(torch.cat([b["img"], b["txt"]], -1)).squeeze(-1)}


class LateFusion(nn.Module):
    """Two unimodal heads; the final score is the mean of their logits."""

    def __init__(self, in_dim=512, d=256, dropout=0.2, **_):
        super().__init__()
        self.h_img, self.h_txt = mlp_head(in_dim, d, dropout), mlp_head(in_dim, d, dropout)

    def forward(self, b):
        return {"logit": 0.5 * (self.h_img(b["img"]) + self.h_txt(b["txt"])).squeeze(-1)}


class HateCLIPper(nn.Module):
    """Cross-fusion from Hate-CLIPper (Kumar & Nandakumar, 2022).

    p_i = W_i x_img, p_t = W_t x_txt (k-dim each); the feature-interaction
    matrix F = p_i p_t^T (k x k) is flattened and fed to an MLP. Every
    image dimension interacts with every text dimension.
    The paper uses CLIP ViT-L/14 with larger k; we use k=128 on ViT-B/32 so all
    methods share the same backbone - see report for this adaptation.
    """

    def __init__(self, in_dim=512, k=128, hidden=512, dropout=0.2, **_):
        super().__init__()
        self.pi = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, k))
        self.pt = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, k))
        self.head = nn.Sequential(nn.Linear(k * k, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, 1))

    def forward(self, b):
        pi = F.normalize(self.pi(b["img"]), dim=-1)
        pt = F.normalize(self.pt(b["txt"]), dim=-1)
        fim = torch.einsum("bi,bj->bij", pi, pt).flatten(1)            # (B,k*k)
        return {"logit": self.head(fim).squeeze(-1)}


# =========================================================================
# Registry + analytic cost model
# =========================================================================
def build_model(name: str, cfg, in_dim: int = 512) -> nn.Module:
    """in_dim = CLIP joint-space width: 512 for ViT-B/32, 768 for ViT-L/14."""
    common = dict(in_dim=in_dim, d=cfg.d_model, dropout=cfg.dropout)
    if name == "image_only":
        return ImageOnlyMLP(**common)
    if name == "text_only":
        return TextOnlyMLP(**common)
    if name == "early_fusion":
        return EarlyFusionMLP(**common)
    if name == "late_fusion":
        return LateFusion(**common)
    if name == "hate_clipper":
        return HateCLIPper(in_dim=in_dim, k=128, hidden=512, dropout=cfg.dropout)
    if name in ("cagco", "cagco_no_ccl"):
        return CAGCoNet(heads=cfg.n_heads, layers=cfg.n_coattn_layers, proj_dim=cfg.proj_dim, **common)
    if name == "cagco_no_coattn":
        return CAGCoNet(heads=cfg.n_heads, layers=cfg.n_coattn_layers, proj_dim=cfg.proj_dim,
                        use_coattn=False, **common)
    raise ValueError(name)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def flops_per_sample(name: str, cfg, n_img=50, n_txt=64, in_dim=512) -> float:
    """Approximate multiply-accumulates (MACs) of the trainable head per meme.
    Linear a->b costs a*b MACs per token. Frozen CLIP cost is shared by all
    models (several GFLOPs per meme, paid once at extraction) and excluded."""
    d = cfg.d_model
    mlp = lambda a: a * d + d
    if name in ("image_only", "text_only"):
        return mlp(in_dim)
    if name == "early_fusion":
        return mlp(2 * in_dim)
    if name == "late_fusion":
        return 2 * mlp(in_dim)
    if name == "hate_clipper":
        k = 128
        return 2 * in_dim * k + k * k + k * k * 512 + 512 * 512 + 512
    base = 2 * in_dim * d + 2 * d + 3 * d * d + d + d * d + d * cfg.proj_dim
    if name == "cagco_no_coattn":
        return base
    proj_in = (n_img + n_txt) * in_dim * d
    per_layer = 0
    for nq, nk in [(n_txt, n_img), (n_img, n_txt)]:
        per_layer += nq * d * d + 2 * nk * d * d + nq * d * d     # Q, K, V, O projections
        per_layer += 2 * nq * nk * d                               # QK^T and AV (all heads)
    per_layer += (n_img + n_txt) * 2 * (d * 2 * d)                 # feed-forward
    pools = (n_img + n_txt) * (d * d + d)
    return base + proj_in + cfg.n_coattn_layers * per_layer + pools
