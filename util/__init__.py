import yaml
from pathlib import Path

# Detector hyperparameters are loaded from config/detectors.yaml so that each
# LVLM's settings can be tuned without editing Python code. Each top-level key
# under `detectors` / `detectors_spatial` matches a `--lvlm` value.
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "detectors.yaml"
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _DETECTOR_CONFIG = yaml.safe_load(_f)

param_dict = {name: dict(params) for name, params in _DETECTOR_CONFIG["detectors"].items()}
param_dict_spatial = {name: dict(params) for name, params in _DETECTOR_CONFIG["detectors_spatial"].items()}

# Prompt templates are loaded from config/questions.yaml (key "prompts").
_QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "config" / "questions.yaml"
with open(_QUESTIONS_PATH, "r", encoding="utf-8") as _f:
    QUESTIONS = yaml.safe_load(_f)["prompts"]