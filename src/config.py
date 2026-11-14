"""
Central configuration for the CAGCo-Net project.

Every script imports CFG from here, so changing a path or a hyper-parameter in
one place changes it everywhere. Values can also be overridden from the command
line of run_all.py (see README).
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ---------------- paths ----------------
    # Folder that holds img/ and the *.jsonl files of the Hateful Memes release
    data_dir: Path = Path("data/hateful_memes")
    out_dir: Path = Path("outputs")            # everything the code writes goes here

    # ---------------- feature extraction ----------------
    clip_name: str = "openai/clip-vit-base-patch32"   # frozen backbone (transfer learning)
    #   switch every CLIP-based model to the larger backbone with:
    #   clip_name = "openai/clip-vit-large-patch14"   (code adapts to 768-d, 257 tokens)
    max_text_tokens: int = 64                  # CLIP allows 77; 64 covers >99% of captions
    extract_batch: int = 64

    # ---------------- confounder recovery ----------------
    hash_size: int = 8                         # dHash -> 64-bit fingerprint
    hash_max_hamming: int = 4                  # <= 4 differing bits = "same image"
    caption_sim: float = 0.90                  # TF-IDF char-ngram cosine >= 0.90 = "same caption"

    # ---------------- model ----------------
    d_model: int = 256                         # width of the fusion space
    n_heads: int = 4                           # co-attention heads
    n_coattn_layers: int = 1                   # bidirectional co-attention blocks
    dropout: float = 0.2
    proj_dim: int = 128                        # contrastive projection head

    # ---------------- confounder-weighted contrastive loss ----------------
    lambda_ccl: float = 0.2                    # weight of L_ccl in the total loss
    tau: float = 0.1                           # temperature
    beta_confounder: float = 3.0               # extra weight on confounder negatives

    # ---------------- training ----------------
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-2
    warmup_frac: float = 0.05
    patience: int = 5                          # early stopping on val AUROC
    seeds: list = field(default_factory=lambda: [0, 1, 2, 3, 4])
    num_workers: int = 0

    # ---------------- fine-tuned benchmarks (proposal: fine-tuned vision model,
    #                  fine-tuned BERT, pretrained VisualBERT) ----------------
    image_backbone: str = "resnet152"          # torchvision ImageNet model (as in Kiela et al.)
    bert_name: str = "bert-base-uncased"
    visualbert_name: str = "uclanlp/visualbert-vqa-coco-pre"   # VisualBERT pretrained on COCO
    n_regions: int = 36                        # Faster R-CNN regions per meme
    region_batch: int = 8
    ft_epochs: int = 5
    ft_batch: int = 32
    ft_lr_backbone: float = 2e-5               # BERT / VisualBERT encoder
    ft_lr_cnn: float = 1e-4                    # ResNet
    ft_lr_head: float = 1e-3                   # new heads / projections
    ft_patience: int = 2
    ft_text_len: int = 64

    # ---------------- evaluation ----------------
    n_bootstrap: int = 2000
    alpha: float = 0.05

    def ensure_dirs(self):
        for sub in ["features", "figures", "runs", "tables", "preds"]:
            (self.out_dir / sub).mkdir(parents=True, exist_ok=True)


CFG = Config()
