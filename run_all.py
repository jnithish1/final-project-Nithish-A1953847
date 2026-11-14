"""
One command to reproduce everything:

    python run_all.py --data-dir data/hateful_memes            # full run: 9 models x 5 seeds
    python run_all.py --data-dir data/hateful_memes --quick    # dry run: 2 seeds, short training
    python run_all.py --out-dir /content/drive/MyDrive/cagco_outputs   # keep results on Drive

Finished (model, seed) runs are skipped when you run it again, so an interrupted Colab
session can simply be restarted with the same command.

Order: EDA on raw data -> CLIP features -> t-SNE -> Faster R-CNN regions ->
       CLIP-feature models (5 incl. CAGCo-Net + ablations) -> fine-tuned benchmarks
       (ResNet-152, BERT, VisualBERT) -> evaluation.
"""
import argparse
import time
from pathlib import Path

from src.config import CFG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-eda", action="store_true")
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--skip-finetuned", action="store_true")
    a = ap.parse_args()
    if a.data_dir: CFG.data_dir = a.data_dir
    if a.out_dir: CFG.out_dir = a.out_dir
    if a.quick:
        CFG.seeds, CFG.epochs, CFG.ft_epochs = [0, 1], 4, 1
    CFG.ensure_dirs()

    t0 = time.time()
    from src import eda, evaluate, extract_features, train
    feats = CFG.out_dir / "features"
    if not a.skip_eda:
        print("\n=== 1/7  EDA on raw data ==="); eda.run_raw_eda()
    if not a.skip_extract and not (feats / "test_txt_pooled.npy").exists():
        print("\n=== 2/7  CLIP feature extraction ==="); extract_features.main()
    if not a.skip_eda:
        print("\n=== 3/7  t-SNE of CLIP features ==="); eda.run_embedding_eda()
    if not a.skip_finetuned and not (feats / "test_region_mask.npy").exists():
        print("\n=== 4/7  Faster R-CNN region features ===")
        from src import extract_regions; extract_regions.main()
    print("\n=== 5/7  Training CLIP-feature models ==="); train.main()
    if not a.skip_finetuned:
        print("\n=== 6/7  Fine-tuning ResNet-152, BERT, VisualBERT ===")
        from src import finetune_baselines; finetune_baselines.main()
    print("\n=== 7/7  Evaluation ==="); evaluate.main()
    print(f"\nAll done in {(time.time()-t0)/60:.1f} min. See {CFG.out_dir}/RESULTS.md and {CFG.out_dir}/figures/")


if __name__ == "__main__":
    main()
