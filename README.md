# Instruction-Lens-Score

PyTorch implementation of InsLen Score (Instruction Lens Score: *Your Instruction Contributes a Powerful Object Hallucination Detector for Multimodal Large Language Models*), ICML 2026.

- Paper: https://arxiv.org/abs/2605.12258
- ICML 2026 poster: https://icml.cc/virtual/2026/poster/62062

> Note: I'm currently busy with the job search, so the rest of the project will be released little by little.

## How to run

```bash
pip install -r requirements.txt
```

### Data paths

All data locations are configurable through `config/datasets.yaml`. Values may
contain `${VAR}` placeholders, which are expanded from the environment;
relative paths are resolved against the repo root. Set:

```bash
export MSCOCO_VAL_DIR=/path/to/MS_COCO2014/val2014   # MSCOCO val2014 images
export OBJECT365_ROOT=/path/to/Object365             # Objects365 refined set
```

`$OBJECT365_ROOT` is expected to contain:

```
val_refined/images/*.jpg
val_refined/annotations_refined.json
object_list_refined.txt
object_list_refined_map.tsv
```

This Objects365 val split was **additionally annotated and re-screened by
hand** (human review on top of an automatic pass), so the per-image ground
truth is more complete than the raw release. Download:

https://www.jianguoyun.com/p/DeDZ6BoQo4jODhjL_agGIAA

extract the archive and point `OBJECT365_ROOT` at its `Object365/` folder.
(You can also just edit the values in `config/datasets.yaml` instead of using
env vars.) Then pick a benchmark with `--dataset`:

```bash
# MSCOCO (default)
python evaluate.py --dataset MSCOCO --lvlm llava-1.5-7b-hf --num_data 300 --seed 0

# Objects365 (refined val set)
python evaluate.py --dataset Objects365 --lvlm llava-1.5-7b-hf --num_data 300 --seed 0
```

Useful flags: `--num_data`, `--seed`, `--max_tokens`, `--inference_temp`,
`--scale`, `-w`. Per-model detector settings live in `config/detectors.yaml`;
per-dataset paths in `config/datasets.yaml`. Results go to `log/`, `figures/`
and `storage/`. The model (`llava-hf/llava-1.5-7b-hf` by default) is downloaded
automatically from Hugging Face.

## Acknowledgements

Thanks to the authors of [GLSIM](https://github.com/deeplearning-wisc/glsim) for open-sourcing their code — part of this implementation is based on it.
