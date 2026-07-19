"""Registry of supported LVLM wrappers.

Only LLaVA-1.5 is enabled for now. To add a new model later: implement a
subclass of ``lvlm.base.BaseLVLM`` in its own module, import it here and add an
entry to ``LVLM_MAP``.
"""

from lvlm.LLaVA import LLaVA

LVLM_MAP = {
    "llava-1.5-7b-hf": LLaVA,
    "llava-1.5-13b-hf": LLaVA,
}
