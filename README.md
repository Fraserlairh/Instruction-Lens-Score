# Instruction-Lens-Score

PyTorch implementation of InsLen Score (Instruction Lens Score: *Your Instruction Contributes a Powerful Object Hallucination Detector for Multimodal Large Language Models*), ICML 2026.

- Paper: https://arxiv.org/abs/2605.12258
- ICML 2026 poster: https://icml.cc/virtual/2026/poster/62062

> Note: I'm currently busy with the job search, so the rest of the project will be released little by little.

## How to run

```bash
pip install -r requirements.txt
```

Before running, set the MSCOCO 2014 val image folder in `evaluate.py` (`MSCOCO_VAL_DIR`).

The model (`llava-hf/llava-1.5-7b-hf` by default) is downloaded automatically from Hugging Face.

```bash
python evaluate.py --lvlm llava-1.5-7b-hf --num_data 300 --seed 0
```
## Acknowledgements

Thanks to the authors of [GLSIM](https://github.com/deeplearning-wisc/glsim) for open-sourcing their code — part of this implementation is based on it.
