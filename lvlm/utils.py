"""Shared helpers for LVLM wrappers and the GL_sim detector.

``find_word`` resolves a detected caption word to a vocabulary token id by
probing a few common surface forms (with/without leading space, plural
suffixes). It is used by :mod:`detector.detector` for models whose tokenizer
does not map words to ids directly, and is kept here (rather than inside a
specific model file) so it survives adding/removing individual model wrappers.
"""

_PREFIXES = ("", " ")
_SUFFIXES = ("", "s", "es")
_INTERN_KEYWORD = "intern"
_ENCODE_MODELS = ("mPLUG_Owl3", "QwenVL3", "llava-one-version1.5", "Pixtral")


def find_word(detected_word, ids, processor, model):
    """Return the token id for ``detected_word`` if it appears in ``ids``.

    Tries a small set of prefix/suffix variants; returns ``None`` when no
    variant tokenizes to a token present in ``ids``.
    """
    for prefix in _PREFIXES:
        for suffix in _SUFFIXES:
            surface = prefix + detected_word + suffix
            token = None
            if _INTERN_KEYWORD in model:
                token = processor(surface, return_tensors="pt")["input_ids"][0, 0]
            elif model in _ENCODE_MODELS:
                encoded = processor.encode(surface)
                if not encoded:
                    continue
                token = encoded[0]
            if token is not None and token in ids:
                return token
    return None
