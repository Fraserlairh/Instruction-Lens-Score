#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Objects365 evaluator for Instruction-Lens-Score.

A drop-in replacement for :class:`util.chair.CHAIR`, but built on the
Objects365 *refined* vocabulary (``object_list_refined.txt``) and the
image-level annotations produced in ``val_refined/annotations_refined.json``,
instead of the MSCOCO 80-class synonym table.

Two surface-form -> canonical tables are combined:

* ``object_list_refined_map.tsv`` -- tab-separated ``alias<TAB>canonical``
  (spelling variants and the cluster merges already decided for the refined
  vocabulary).
* an MSCOCO-CHAIR style synonym table (``object_365_synonyms.tsv``) --
  ``canonical, syn1, syn2, ...`` per line, where the first name is the
  canonical class.

The original detector only requires the evaluator object to expose
``compute_hallucinations(img_id, caption, args)`` returning a dict with the
keys ``recall_idxs`` / ``hallucination_idxs`` (plus the ``*_words`` lists used
by the special-tokenizer models).  This class provides exactly that, so it can
be passed straight to ``detector.detector.compute_scores`` without touching any
of the original files.
"""

import json
import os

import nltk
import inflect

_p = inflect.engine()


def singularize(word):
    """Best-effort singularisation, mirroring util.chair.singularize."""
    return _p.singular_noun(word) or word


# Colour adjectives (a word here is treated as a colour, not an object, when it
# is used attributively / in a colour enumeration). Classes that are *also*
# colour words are guarded by context rules instead of being dropped outright,
# so the genuine fruit/vegetable sense is kept ("a bowl of oranges").
COLORS = {
    "black", "white", "red", "green", "blue", "yellow", "orange", "pink",
    "purple", "grey", "gray", "brown", "beige", "tan", "gold", "silver",
    "violet", "indigo", "cyan", "magenta", "maroon", "navy", "teal",
    "turquoise", "lavender", "crimson", "khaki", "ivory", "olive", "peach",
    "coral", "salmon", "amber", "bronze", "burgundy", "cream", "scarlet",
}
# Class names that double as colour words -> apply context guard on these.
COLOR_CLASSES = {
    "Orange", "Peach", "Lemon", "Plum", "Cherry", "Grape", "Tomato",
    "Eggplant", "Avocado", "Pumpkin", "Nuts",
}
_DETERMINERS = {
    "a", "an", "the", "this", "that", "these", "those", "some", "any", "many",
    "several", "few", "one", "two", "three", "four", "five", "no", "all",
    "both", "each", "every",
}
_STOP = {
    "and", "or", ",", ".", ";", ":", "of", "with", "on", "in", "at", "to",
    "for", "from", "by", "is", "are", "was", "were", "be", "as", "it",
}
_COLOR_VERBS = {"is", "are", "was", "were", "look", "looks", "appear",
                "appears", "colored", "coloured", "colour", "color", "painted"}


def is_color_usage(tokens, i):
    """Heuristic: is ``tokens[i]`` used as a colour rather than a noun?"""
    n = len(tokens)
    prev = tokens[i - 1] if i > 0 else None
    nxt = tokens[i + 1] if i + 1 < n else None
    nxt2 = tokens[i + 2] if i + 2 < n else None
    # coordinated with another colour: "orange and red", "red , orange"
    if prev in COLORS or nxt in COLORS:
        return True
    if nxt in ("and", "or", ",") and nxt2 in COLORS:
        return True
    # predicative: "the wall is orange"
    if prev in _COLOR_VERBS:
        return True
    # attributive: "orange fish" (colour + noun, no preceding determiner)
    if (nxt and nxt.isalpha() and nxt not in _STOP and nxt not in COLORS
            and prev not in _DETERMINERS):
        return True
    return False


class Object365CHAIR:
    """Image-level Objects365 ground truth + caption word matcher.

    Parameters
    ----------
    annotations_path:
        ``annotations_refined.json`` produced by ``refine_annotations.py``.
    vocab_path:
        ``object_list_refined.txt`` -- one canonical class name per line.
    map_path:
        ``object_list_refined_map.tsv`` -- raw/alias name -> canonical name.
    gt_key:
        Which per-image field to use as ground truth.  Defaults to
        ``objects_all_strict`` ("评测建议使用").
    synonyms_path:
        MSCOCO-CHAIR style synonym table (``canonical, syn1, syn2, ...``).
    """

    def __init__(self, annotations_path, vocab_path, map_path,
                 gt_key="objects_all_strict", synonyms_path=None):
        # ---- vocabulary + alias/synonym tables --------------------------
        with open(vocab_path, encoding="utf-8") as f:
            self.vocab = [l.strip() for l in f if l.strip()]
        self.vocab_set = set(self.vocab)

        # Surface form -> canonical class. The refined-vocab map is loaded
        # first (spelling variants + cluster merges); the MSCOCO-style synonym
        # table is layered on top WITHOUT overriding it (setdefault).
        alias = {}
        if map_path and os.path.exists(map_path):
            for raw, canon in self._load_map(map_path).items():
                alias.setdefault(raw, canon)
        if synonyms_path and os.path.exists(synonyms_path):
            for raw, canon in self._load_synonyms(synonyms_path).items():
                alias.setdefault(raw, canon)

        # phrase keys (both the raw-lower form and its singularised form) ->
        # canonical class. Keeping *both* keys avoids the inflect quirk where
        # singularise("glass") == "glas" but singularise("glasses") == "glass",
        # which would otherwise stop "wine glasses" from matching "Wine Glass".
        self.phrase_to_canon = {}

        # Plural-only nouns whose singular would collide with another common
        # word ("glasses" -> "glass" the material); do not register the
        # singular key for these.
        plural_only = {"glasses"}

        def add(phrase, canon):
            keys = self._norm(phrase)
            if phrase.strip().lower() in plural_only:
                keys = (tuple(w.lower() for w in phrase.split()),)
            for key in keys:
                self.phrase_to_canon.setdefault(key, canon)

        for name in self.vocab:
            add(name, name)
        for raw, canon in alias.items():
            if canon in self.vocab_set:
                add(raw, canon)
        self.max_phrase_len = max(len(k) for k in self.phrase_to_canon)

        # ---- ground truth: image_id -> set(objects) ----------------------
        with open(annotations_path, encoding="utf-8") as f:
            data = json.load(f)
        self.gt_key = gt_key
        self.imid_to_objects = {}
        for rec in data["images"]:
            objs = rec.get(gt_key)
            if objs is None:
                objs = rec.get("objects", [])
            self.imid_to_objects[rec["image_id"]] = set(objs)

    @staticmethod
    def _load_map(path):
        """Tab-separated ``alias<TAB>canonical`` table (object_list_refined_map.tsv)."""
        table = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    raw, canon = parts[0].strip(), parts[1].strip()
                    if raw and canon:
                        table[raw] = canon
        return table

    @staticmethod
    def _load_synonyms(path):
        """MSCOCO-CHAIR style table: ``canonical, syn1, syn2, ...`` per line.

        The first comma-separated name is the canonical class; every name after
        it is a synonym that maps onto it.
        """
        table = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = [p.strip() for p in line.rstrip("\n").split(",")]
                parts = [p for p in parts if p]
                if len(parts) >= 2:
                    canon = parts[0]
                    for syn in parts[1:]:
                        table.setdefault(syn, canon)
        return table

    @staticmethod
    def _norm(phrase):
        """Matching keys for a phrase: its raw-lower and singularised tuples."""
        toks = [w.lower() for w in phrase.split()]
        raw = tuple(toks)
        sing = tuple(singularize(t) for t in toks)
        return (raw,) if sing == raw else (raw, sing)

    def caption_to_words(self, caption):
        """Locate Objects365 objects mentioned in ``caption``.

        Returns ``(words, node_words, idxs, tokens)`` mirroring
        ``util.chair.CHAIR.caption_to_words``:

        * ``tokens``     -- plain ``nltk.word_tokenize(caption.lower())``
        * ``words``      -- matched surface phrases (as they appear)
        * ``node_words`` -- the canonical class each match maps to
        * ``idxs``       -- index of the *first* token of each match, so that
          ``tokens[idxs]`` reproduces the surface word used by the detector.
        """
        tokens = nltk.word_tokenize(caption.lower())

        words, node_words, idxs = [], [], []
        i, n = 0, len(tokens)
        while i < n:
            matched = False
            upper = min(self.max_phrase_len, n - i)
            # greedy longest match first (max_len -> ... -> 1 words)
            for length in range(upper, 0, -1):
                window = tokens[i:i + length]
                key_raw = tuple(window)
                key_sing = tuple(singularize(w) for w in window)
                canon = self.phrase_to_canon.get(key_raw)
                if canon is None and key_sing != key_raw:
                    canon = self.phrase_to_canon.get(key_sing)
                if canon is not None:
                    # Colour guard: drop an ambiguous colour word only when it
                    # is actually used as a colour here.
                    if (canon in COLOR_CLASSES and length == 1
                            and is_color_usage(tokens, i)):
                        i += 1
                        matched = True
                        break
                    words.append(" ".join(window))
                    node_words.append(canon)
                    idxs.append(i)
                    i += length
                    matched = True
                    break
            if not matched:
                i += 1
        return words, node_words, idxs, tokens

    def compute_hallucinations(self, imid, cap, args=None):
        """Split caption object mentions into recalled vs hallucinated ones."""
        words, node_words, idxs, _ = self.caption_to_words(cap)
        gt_objects = self.imid_to_objects.get(imid, set())

        cap_dict = {
            "mscoco_hallucinated_words": [],
            "mscoco_gt_words": list(gt_objects),
            "mscoco_generated_words": list(node_words),
            "hallucination_idxs": [],
            "hallucinated_words": 0,
            "recall_words": [],
            "recall_idxs": [],
        }

        for word, node_word, idx in zip(words, node_words, idxs):
            if node_word not in gt_objects:
                cap_dict["hallucinated_words"] += 1
                cap_dict["mscoco_hallucinated_words"].append((word, node_word))
                cap_dict["hallucination_idxs"].append(idx)
            else:
                cap_dict["recall_words"].append((word, node_word))
                cap_dict["recall_idxs"].append(idx)

        return cap_dict
