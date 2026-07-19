"""Abstract interface shared by all supported LVLM wrappers.

Every concrete model wrapper (e.g. ``LLaVA``) is expected to subclass
``BaseLVLM`` and implement :meth:`build_model` and :meth:`generate`.

The evaluation pipeline (``evaluate.py``) only relies on this contract:

* ``model = ConcreteLVLM(args)`` loads the weights.
* ``model.generate(image, question, img_id, args)`` returns a dict whose keys
  are consumed by ``detector.detector.compute_scores``.

Keeping the contract in one place makes it trivial to add new models later:
implement a subclass, register it in ``lvlm/__init__.py`` and nothing else in
the pipeline needs to change.

Set the environment variable ``LVLM_DEBUG=1`` to have every subclass'
``generate`` output automatically validated against ``GENERATE_OUTPUT_KEYS``;
this is a no-op (zero overhead) when the flag is off.
"""

import os
import functools
import warnings
from abc import ABC, abstractmethod

# Keys that :meth:`BaseLVLM.generate` must return. Documented here so new model
# wrappers can be validated against a single source of truth.
GENERATE_OUTPUT_KEYS = (
    "instruction_tokens",   # List[str]
    "hidden_states",        # raw ``output.hidden_states`` tuple
    "output_ids",           # LongTensor [len]
    "image_start",          # int
    "image_end",            # int
    "answer_start",         # int
    "tokens",               # List[str]
    "final_ans",            # str
    "img_id",               # image id
    "vq_hidden_states",     # Tensor [layer, 1, token_len, dim]
    "answer_hidden_states",  # Tensor [token_len, layer, dim]
    "answer_attentions",    # List[Tensor [layer, head, 1, seq_len]]
)


def _debug_enabled():
    return os.environ.get("LVLM_DEBUG", "0").lower() not in ("", "0", "false")


class BaseLVLM(ABC):
    """Common base class for large vision-language model wrappers."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # In debug mode, wrap the subclass' ``generate`` so its output dict is
        # validated against the contract on every call.
        if _debug_enabled() and "generate" in cls.__dict__:
            original = cls.generate

            @functools.wraps(original)
            def _checked_generate(self, *args, **kwargs):
                output = original(self, *args, **kwargs)
                self.validate_generate_output(output)
                return output

            cls.generate = _checked_generate

    def __init__(self, args):
        self.version = args.lvlm
        self.build_model(args)

    @abstractmethod
    def build_model(self, args):
        """Load the model, processor / tokenizer and cache token ids."""

    @abstractmethod
    def generate(self, image, question, img_id, args):
        """Run generation and return a dict keyed by ``GENERATE_OUTPUT_KEYS``."""

    def easy_generate(self, image, question, args, mask_idx=None):
        """Lightweight generation used for masking ablations (optional)."""
        raise NotImplementedError

    @staticmethod
    def validate_generate_output(output):
        """Check that ``generate`` returned a dict with the expected keys.

        Raises ``TypeError`` / ``KeyError`` on a missing key; warns (rather than
        fails) on unexpected extra keys so wrappers may return auxiliary data.
        """
        if not isinstance(output, dict):
            raise TypeError(
                f"generate() must return a dict, got {type(output).__name__}"
            )
        missing = [key for key in GENERATE_OUTPUT_KEYS if key not in output]
        if missing:
            raise KeyError(f"generate() output missing required keys: {missing}")
        extra = [key for key in output if key not in GENERATE_OUTPUT_KEYS]
        if extra:
            warnings.warn(f"generate() output has unexpected extra keys: {extra}")
        return output
