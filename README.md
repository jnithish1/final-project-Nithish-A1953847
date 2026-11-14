# CAGCo-Net: Confounder-Aware Gated Co-Attention for Hateful Meme Detection

Capstone project code: a multimodal (image + caption) classifier for the Hateful Memes
dataset, compared against the five benchmarks of the proposal (fine-tuned image-only ResNet-152,
fine-tuned text-only BERT, early/late fusion, VisualBERT, Hate-CLIPper) and two ablations,
with full EDA, statistics and complexity analysis.

## 1. What is in this folder

```
cagconet/
├── README.md               this file
├── requirements.txt        Python packages
├── run_all.py              one command that runs the whole pipeline
├── run_in_colab.ipynb      the same pipeline as a Colab notebook (GPU)
└── src/
    ├── config.py           every path and hyper-parameter in one place
    ├── data.py             loads the jsonl splits, dHash (from scratch), confounder-pair recovery
    ├── eda.py              Criterion 1 figures + statistics
    ├── extract_features.py frozen CLIP ViT-B/32 features, cached once (transfer learning)
    ├── extract_regions.py  36 Faster R-CNN region features per meme (input for VisualBERT)
    ├── models.py           CAGCo-Net, early/late fusion, Hate-CLIPper - from scratch (no nn.MultiheadAttention)
    ├── finetune_baselines.py  fine-tuned ResNet-152 (image-only), BERT (text-only), VisualBERT
    ├── losses.py           BCE + confounder-weighted supervised contrastive loss
    ├── train.py            training loop, early stopping, cost measurement
    ├── stats_tests.py      DeLong, paired bootstrap, McNemar, paired t / Wilcoxon, Holm
    ├── evaluate.py         metrics, significance tables, all result figures, RESULTS.md
    └── smoke_test.py       self-check of every model on random data (no dataset needed)
```

## 2. Get the data

The Hateful Memes dataset (Kiela et al., 2020) is released by Meta AI under its own research
licence. The quickest way to get the original files is the Hugging Face mirror, which keeps the
original layout (about 3.4 GB):

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="neuralcatcher/hateful_memes", repo_type="dataset",
                  local_dir="data/hateful_memes", max_workers=8)
```

The official source is the DrivenData "Hateful Memes" competition (https://hatefulmemeschallenge.com).
Either way, read LICENSE.txt. The folder must look like this:

```
data/hateful_memes/
├── img/                 10,000 .png memes
├── train.jsonl          8,500 labelled
├── dev_seen.jsonl       500 labelled   (some mirrors call it dev.jsonl)
├── dev_unseen.jsonl     540 labelled   (optional)
├── test_seen.jsonl      1,000          (labelled in the phase-2 release; optional)
└── test_unseen.jsonl    2,000          (optional, not used)
```

Split logic (automatic, see `data.load_splits`):

| Files found | Validation | Test |
|---|---|---|
| labelled `test_seen.jsonl` | dev_seen | test_seen |
| `dev_unseen.jsonl` | dev_unseen minus ids shared with dev_seen | dev_seen |
| only train + dev | 10% of train, split by confounder group | dev_seen |

## 3. Install and run

```bash
git clone https://github.com/jnithish1/final-project-Nithish-A1953847.git
cd final-project-Nithish-A1953847
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt

python -m src.smoke_test                                # ~1 min, CLIP-feature models, no downloads
python -m src.smoke_test --finetuned                    # + ResNet-152, BERT, Faster R-CNN, VisualBERT
python run_all.py --data-dir data/hateful_memes --quick # dry run: 2 seeds, short training
python run_all.py --data-dir data/hateful_memes         # full run: 9 models x 5 seeds
```

Step by step (what `run_all.py` calls):

```bash
python -m src.eda                  # raw-data EDA            -> outputs/figures/eda_*.png
python -m src.extract_features     # CLIP features (once)    -> outputs/features/*.npy  (~1.2 GB)
python -m src.eda --embeddings     # t-SNE of CLIP features  -> outputs/figures/eda_clip_tsne.png
python -m src.extract_regions      # Faster R-CNN regions    -> outputs/features/*_regions.npy (~0.8 GB)
python -m src.train                # CLIP-feature models     -> outputs/preds, outputs/runs
python -m src.train --models cagco --seeds 0 1   # or a subset
python -m src.finetune_baselines   # ResNet-152, BERT, VisualBERT fine-tuning
python -m src.evaluate             # tables, tests, figures  -> outputs/RESULTS.md
```

A GPU is needed. The fine-tuned benchmarks are the slow part: ResNet-152, BERT and VisualBERT
are trained end to end for 5 seeds each, which takes a few hours on a Colab T4. Finished
(model, seed) runs are skipped on a re-run, so if Colab disconnects just run the same command
again. Put `--out-dir` on Google Drive so results survive a disconnect. To save time you can set
`image_backbone = "resnet50"` in `config.py` or pass `--seeds 0 1 2` to `finetune_baselines`.

## 4. Outputs

| File | What it holds |
|---|---|
| `outputs/RESULTS.md` | every results table in markdown, ready to paste into the report |
| `outputs/tables/metrics_summary.csv` | AUROC, accuracy, precision, recall, macro-F1: mean, std, 95% CI |
| `outputs/tables/significance.csv` | CAGCo-Net vs each model: paired t, Wilcoxon, DeLong, bootstrap, McNemar, Holm |
| `outputs/tables/complexity.csv` | parameters, MACs per meme, seconds per epoch, inference ms, peak GPU MB |
| `outputs/tables/category_accuracy.csv` | accuracy on recovered confounder categories |
| `outputs/tables/eda_summary.json` | numbers behind the EDA figures |
| `outputs/figures/eda_*.png` | class balance, caption length, image properties, log-odds words, confounders, t-SNE |
| `outputs/figures/res_*.png` | ROC, AUROC CI, confusion matrices, learning curves, bootstrap, gate, attention |

## 5. Method in one paragraph

Benchmarks follow the proposal exactly: an image-only ResNet-152 and a text-only BERT, both
fine-tuned; early and late fusion of CLIP features; VisualBERT pretrained on COCO and fine-tuned
on caption tokens + Faster R-CNN regions; and Hate-CLIPper's outer-product interaction on the
same frozen CLIP features. Frozen CLIP ViT-B/32 turns each meme into 50 image tokens and up to 64 caption tokens in a
shared 512-d space. CAGCo-Net projects both to 256-d, runs a bidirectional co-attention block
(caption words attend to image patches and vice versa), attention-pools each stream, adds the
pooled CLIP embedding back as a residual, and mixes the two streams with a learned per-meme gate
g. The fused vector `[g·h_img ; (1-g)·h_txt ; h_img ⊙ h_txt]` feeds an MLP classifier. Training
adds a confounder-weighted supervised contrastive loss: memes that share an image but not a
caption (or the reverse) and have opposite labels are recovered with a from-scratch perceptual
hash, kept in the same mini-batch, and weighted β times harder as negatives.

## 6. Known limitations and possible bugs

* Confounder pairs are recovered approximately, as stated in the proposal (dHash ≤ 4 bits for
  "same image", TF-IDF character-n-gram cosine ≥ 0.90 for "same caption"). Meme
  images contain the rendered caption, so two "same image" memes differ in the text region;
  a threshold that is too tight misses pairs, too loose creates false pairs. Check
  `outputs/tables/confounder_pairs_train.csv` and `eda_confounder_examples.png`.
* Captions longer than 64 CLIP tokens are truncated (the extractor prints how many).
* All features sit in GPU memory (~1.2 GB fp16 for train). On a small GPU, edit
  `FeatureSplit` to keep them on CPU.
* Hate-CLIPper uses the proposal's shared backbone, CLIP ViT-B/32 (k = 128). The published 85.8
  AUROC used ViT-L/14. To run every CLIP-based model on ViT-L/14, set
  `clip_name = "openai/clip-vit-large-patch14"` in `config.py` and delete `outputs/features`.
* VisualBERT's region features come from torchvision's COCO Faster R-CNN (1024-d box-head
  vectors, linearly projected to VisualBERT's 2048-d input), not the Visual Genome detector of
  the original MMF baseline, which needs detectron2.
* The sub-categories (multimodal hate, benign text/image confounder, ...) are mapped from the
  recovered pairs; unimodal hate cannot be told apart from unpaired multimodal hate.
* The DeLong and McNemar tests treat test memes as independent; confounder pairs in the test set
  slightly violate this.
* With 5 seeds the smallest possible two-sided Wilcoxon p-value is 0.0625, so the paired t-test
  is the main across-seed test.
* `train.py` uses `torch.backends.cudnn.deterministic = True`; results are repeatable on the same
  GPU and library versions, not bit-identical across machines.
