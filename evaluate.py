"""Evaluate object-hallucination detection.

This entry point loads a dataset, runs a target LVLM to generate captions,
applies the GL_sim detector and reports detection performance (AUROC / AUPR /
FPR@95%TPR) for the various score types.
"""

import os
import sys
import time
import random
from pathlib import Path
from datetime import datetime

import json
import pickle
import warnings
from argparse import ArgumentParser

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from util.measuring import get_measures
from detector.detector import compute_scores
from lvlm import LVLM_MAP
from util.chair import CHAIR
from util import param_dict, QUESTIONS

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths (environment specific; override here for a new machine).
# ---------------------------------------------------------------------------
MSCOCO_VAL_DIR = "/data/yanqi/Storage/MS_COCO2014/val2014"
MSCOCO_ANNOTATION_PATH = "coco_ground_truth.json"
CHAIR_CACHE_PATH = "chair.pkl"


class _CHAIRUnpickler(pickle.Unpickler):
    """Load a CHAIR instance regardless of the module it was pickled from."""

    def find_class(self, module, name):
        if name == "CHAIR":
            return CHAIR
        return super().find_class(module, name)


def load_chair_evaluator(path):
    with open(path, "rb") as f:
        return _CHAIRUnpickler(f).load()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--lvlm", type=str, default="llava-1.5-7b-hf",
        choices=list(LVLM_MAP.keys()),
    )
    parser.add_argument("--dataset", type=str, default="MSCOCO", choices=["MSCOCO"])
    parser.add_argument("--inference_temp", type=float, default=0.1)
    parser.add_argument("--sampling_temp", type=float, default=1.0)
    parser.add_argument("--sampling_time", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--generate", type=bool, default=True)
    parser.add_argument("--num_data", type=int, default=300)
    parser.add_argument("--w", type=float, default=0.5)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def obtain_lvlm(args):
    lvlm_class = LVLM_MAP.get(args.lvlm)
    if lvlm_class is None:
        raise ValueError(f"Unsupported LVLM: {args.lvlm}")
    return lvlm_class(args)


def fix_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_or_save_pkl(args, path, coco_data):
    path = Path(path)
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    coco_gt = random.sample(coco_data, args.num_data)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(coco_gt, f)
    return coco_gt


def plot_density(true_scores, false_scores, name):
    plt.clf()
    plt.figure(figsize=(8, 6))
    sns.kdeplot(true_scores, label="Truth", fill=True, alpha=0.3)
    sns.kdeplot(false_scores, label="Hallucination", fill=True, alpha=0.3)
    plt.xlabel("Confidence")
    plt.ylabel("Density")
    plt.title("Density Comparison")
    plt.legend()
    plt.savefig("figures/" + name)


def _fmt_ts(t):
    """Format a unix timestamp as a human-readable local datetime string."""
    return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")


def _write_metric_header(f, args, run_start):
    """Write the metadata header + column titles of the metric report."""
    cmd = " ".join(sys.argv)
    f.write("=" * 64 + "\n")
    f.write("GL_sim Evaluation Report\n")
    f.write("=" * 64 + "\n")
    meta = [
        ("lvlm", args.lvlm),
        ("dataset", args.dataset),
        ("seed", args.seed),
        ("num_data", args.num_data),
        ("w (gl weight)", args.w),
        ("temperature", 0.01),
        ("command", cmd),
        ("start_time", _fmt_ts(run_start)),
    ]
    for key, val in meta:
        f.write(f"{key:<16}: {val}\n")
    f.write("-" * 64 + "\n")
    f.write(f"{'metric':<16} | {'AUROC':>12} | {'AUPR':>12} | {'FPR@95%TPR':>12}\n")
    f.write("-" * 64 + "\n")
    f.flush()


def _write_metric_row(f, name, auroc, aupr, fpr):
    """Write a single metric result row, aligned and rounded to 4 decimals."""
    f.write(f"{name:<16} | {auroc:>12.4f} | {aupr:>12.4f} | {fpr:>12.4f}\n")
    f.flush()


def main():
    args = parse_args()
    fix_seed(args.seed)

    ms_coco_val_dir = MSCOCO_VAL_DIR
    annotation_path = MSCOCO_ANNOTATION_PATH
    evaluator = load_chair_evaluator(CHAIR_CACHE_PATH)
    with open(annotation_path, "r") as f:
        coco_data = [json.loads(line) for line in f]
    args.evaluator = evaluator

    question = QUESTIONS["prompt_o"]

    coco_gt = read_or_save_pkl(args, f"data/{args.dataset}_data_{args.num_data}_{args.seed}.pkl", coco_data)

    lvlm = obtain_lvlm(args)

    # Accumulators for every score type GL_sim produces (recalled / hallucinated).
    accumulators = {key: [] for key in (
        "global_cos_matrix_true", "global_cos_matrix_false",
        "top_k_cos_matrix_true", "top_k_cos_matrix_false",
        "calibrated_local_true", "calibrated_local_false",
        "context_consistency_true", "context_consistency_false",


        "mean_prob_matrix_true", "mean_prob_matrix_false",
        "svar_true", "svar_false",
    )}

    print(param_dict[args.lvlm])

    # ---- metric log setup -------------------------------------------------
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    metric_path = f"log/{args.dataset}_{args.lvlm}_{args.seed}_{run_stamp}.txt"
    run_start = time.time()
    metric_file = open(metric_path, "w")
    _write_metric_header(metric_file, args, run_start)

    for i, entry in enumerate(tqdm(coco_gt, desc="Processing Images")):
        args.current_id = entry["image"]
        image_filename = entry["image"]
        image_path = os.path.join(ms_coco_val_dir, image_filename)

        if not os.path.exists(image_path):
            print(f"Warning: Image {image_filename} not found. Skipping.")
            continue

        image = Image.open(image_path).convert("RGB")

        result = lvlm.generate(image, question, entry["image_id"], args)

        scores = compute_scores(
            args, result["vq_hidden_states"], result["answer_hidden_states"],
            result["output_ids"], result["image_start"], result["image_end"],
            result["answer_start"], result["tokens"], result["final_ans"], result["img_id"],
            lvlm, args.evaluator, result["answer_attentions"],
            **param_dict[args.lvlm],
        )

        for key, value in scores.items():
            accumulators[key] += value

        suffix = f"_temp_0.01_{args.lvlm}_{args.dataset}"
        if (i + 1) % 10 == 0:
            if not accumulators["mean_prob_matrix_true"] or not accumulators["mean_prob_matrix_false"]:
                continue

            stacked = {
                key: torch.cat(accumulators[key], dim=0).detach().to(torch.float32).cpu().numpy()
                for key in accumulators
            }

            auroc, aupr, fpr = get_measures(stacked["mean_prob_matrix_true"], stacked["mean_prob_matrix_false"])
            print(f"{'[Calibration Confidence]':<26} AUROC: {auroc:>9.4f} | AUPR: {aupr:>9.4f} | FPR@95% TPR: {fpr:>9.4f}")
            if (i + 1) == args.num_data:
                _write_metric_row(metric_file, "Calibration Confidence", auroc, aupr, fpr)

            auroc, aupr, fpr = get_measures(stacked["context_consistency_true"], stacked["context_consistency_false"])
            print(f"{'[Context Consistency]':<26} AUROC: {auroc:>9.4f} | AUPR: {aupr:>9.4f} | FPR@95% TPR: {fpr:>9.4f}")
            plot_density(stacked["context_consistency_true"], stacked["context_consistency_false"], f"density_context_consistency{suffix}.jpeg")
            np.save(f"storage/context_consistency_result{suffix}.npy",
                    {"true_scores": stacked["context_consistency_true"], "false_scores": stacked["context_consistency_false"]})
            if (i + 1) == args.num_data:
                _write_metric_row(metric_file, "Context Consistency", auroc, aupr, fpr)

            auroc, aupr, fpr = get_measures(stacked["global_cos_matrix_true"], stacked["global_cos_matrix_false"])
            print(f"{'[Global Score]':<26} AUROC: {auroc:>9.4f} | AUPR: {aupr:>9.4f} | FPR@95% TPR: {fpr:>9.4f}")
            np.save(f"storage/global_score_result{suffix}.npy",
                    {"true_scores": stacked["global_cos_matrix_true"], "false_scores": stacked["global_cos_matrix_false"]})
            if (i + 1) == args.num_data:
                _write_metric_row(metric_file, "Global Score", auroc, aupr, fpr)

            auroc, aupr, fpr = get_measures(stacked["top_k_cos_matrix_true"], stacked["top_k_cos_matrix_false"])
            print(f"{'[Local Score]':<26} AUROC: {auroc:>9.4f} | AUPR: {aupr:>9.4f} | FPR@95% TPR: {fpr:>9.4f}")
            np.save(f"storage/local_score_result{suffix}.npy",
                    {"true_scores": stacked["top_k_cos_matrix_true"], "false_scores": stacked["top_k_cos_matrix_false"]})
            if (i + 1) == args.num_data:
                _write_metric_row(metric_file, "Local Score", auroc, aupr, fpr)

            auroc, aupr, fpr = get_measures(
                stacked["calibrated_local_true"], stacked["calibrated_local_false"])
            print(f"{'[Calibrated Local]':<26} AUROC: {auroc:>9.4f} | AUPR: {aupr:>9.4f} | FPR@95% TPR: {fpr:>9.4f}")
            np.save(f"storage/calibrated_local_result{suffix}.npy",
                    {"true_scores": stacked["calibrated_local_true"],
                     "false_scores": stacked["calibrated_local_false"]})
            if (i + 1) == args.num_data:
                _write_metric_row(metric_file, "calibrated local", auroc, aupr, fpr)

            auroc, aupr, fpr = get_measures(
                args.w * stacked["global_cos_matrix_true"] + (1 - args.w) * stacked["top_k_cos_matrix_true"],
                args.w * stacked["global_cos_matrix_false"] + (1 - args.w) * stacked["top_k_cos_matrix_false"],
            )
            print(f"{'[GLSIM]':<26} AUROC: {auroc:>9.4f} | AUPR: {aupr:>9.4f} | FPR@95% TPR: {fpr:>9.4f}")
            if (i + 1) == args.num_data:
                _write_metric_row(metric_file, "GLSIM", auroc, aupr, fpr)

            w = 0.4
            auroc, aupr, fpr = get_measures(
                (1 - w) * stacked["calibrated_local_true"] + w * stacked["context_consistency_true"],
                (1 - w) * stacked["calibrated_local_false"] + w * stacked["context_consistency_false"],
            )
            print(f"{'[Instruction Lens Score]':<26} AUROC: {auroc:>9.4f} | AUPR: {aupr:>9.4f} | FPR@95% TPR: {fpr:>9.4f}")
            if (i + 1) == args.num_data:
                _write_metric_row(metric_file, "Instruction Lens Score", auroc, aupr, fpr)

            auroc, aupr, fpr = get_measures(stacked["svar_true"], stacked["svar_false"])
            print(f"{'[SVAR]':<26} AUROC: {auroc:>9.4f} | AUPR: {aupr:>9.4f} | FPR@95% TPR: {fpr:>9.4f}")
            np.save(f"storage/svar_result{suffix}.npy",
                    {"true_scores": stacked["svar_true"], "false_scores": stacked["svar_false"]})
            if (i + 1) == args.num_data:
                _write_metric_row(metric_file, "SVAR", auroc, aupr, fpr)

    # ---- metric log footer (timing) --------------------------------------
    run_end = time.time()
    metric_file.write("-" * 64 + "\n")
    metric_file.write(f"{'end_time':<16}: {_fmt_ts(run_end)}\n")
    metric_file.write(f"{'elapsed':<16}: {run_end - run_start:.1f} s\n")
    metric_file.write("=" * 64 + "\n")
    metric_file.close()


if __name__ == "__main__":
    main()
