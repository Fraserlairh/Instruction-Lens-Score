import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from lvlm.utils import find_word

# Models whose tokenizer needs the special `find_word` lookup instead of the
# default tokenizer based token-id resolution.
SPECIAL_TOKENIZER_MODELS = (
    "internVL",
    "mPLUG_Owl3",
    "QwenVL3",
    "llava-one-version1.5",
    "Pixtral",
)
COMPLEMENT_WORDS = ("person", "people")


def compute_svar(attn, image_start, image_end, layer_list):
    """Sum attention mass over the image region across the given layers."""
    if attn == []:
        return 0
    attn = attn[:, :, :, image_start:image_end].sum(dim=-1)
    attn = attn[:, :, 0].sum(dim=-1)
    return sum(attn[layer] for layer in layer_list)


def _resolve_token(args, cap_dict, target_indexes, idx, true_idx, tokens, output_ids_rs, model):
    """Resolve the vocabulary token id that corresponds to a target word.

    Returns ``(token, detected_word)`` or ``(None, detected_word)`` when the
    token cannot be found (caller should skip in that case).
    """
    if args.lvlm in SPECIAL_TOKENIZER_MODELS:
        detected_word = cap_dict[_WORD_KEY_DICT[target_indexes]][idx][0]
        token = None
        if detected_word in COMPLEMENT_WORDS:
            for element in COMPLEMENT_WORDS:
                if "intern" in args.lvlm:
                    token = find_word(element, output_ids_rs, model.processor, args.lvlm)
                if "mPLUG" in args.lvlm or "Qwen" in args.lvlm or "llava-one" in args.lvlm:
                    token = find_word(element, output_ids_rs, model.tokenizer, args.lvlm)
                if token is not None:
                    break
        else:
            if "intern" in args.lvlm:
                token = find_word(detected_word, output_ids_rs, model.processor, args.lvlm)
            if args.lvlm in ("mPLUG_Owl3", "QwenVL3", "llava-one-version1.5", "Pixtral"):
                token = find_word(detected_word, output_ids_rs, model.tokenizer, args.lvlm)
        return token, detected_word

    inputs = model.processor.tokenizer(
        tokens[true_idx], return_tensors="pt", add_special_tokens=False
    ).input_ids
    return inputs[0, 0], tokens[true_idx]


_WORD_KEY_DICT = {
    "recall_idxs": "recall_words",
    "hallucination_idxs": "mscoco_hallucinated_words",
}
_COLLECTED_INDEXES = ("recall_idxs", "hallucination_idxs")


@dataclass
class WordState:
    """Precomputed per-word tensors shared by all detectors for one word.

    Built once per target word by :func:`_build_word_state`; each
    :func:`score_*` detector then reads the slices it needs. This avoids
    recomputing the shared Global / Top-k / Context quantities for every
    detector while keeping each detector self-contained.
    """

    token: int
    w: torch.Tensor                       # max question prob for this token
    detected_embeddings: torch.Tensor     # answer_hidden_states[toke_idx - 1]
    detected_embeddings_norm: torch.Tensor
    global_cos_matrix: torch.Tensor       # Global detector raw score (GLSIM family)
    top_k_cos_matrix: torch.Tensor        # Top-k / Local detector raw score (GLSIM family)
    ils_top_k_cos_matrix: torch.Tensor    # Top-k / Local detector raw score (ILS family, decoupled)
    mean_context_prob: torch.Tensor       # context-family shared probability
    weighed_diff: torch.Tensor            # context-family shared difference
    svar: torch.Tensor                    # SVAR detector raw score


# ---------------------------------------------------------------------------
# Individual detectors. Each consumes a WordState (and the scalar `bias` where
# needed) and returns the score tensor for one word; the dispatcher appends it
# to the matching "*_true" / "*_false" matrix.
# ---------------------------------------------------------------------------

def score_global(state):
    """Global alignment between the final instruction embedding and the word."""
    return state.global_cos_matrix.reshape([1, 1])


def score_top_k(state):
    """Local (Top-k) cosine similarity between the word and image patches."""
    return state.top_k_cos_matrix


def calibrated_local_score(state):
    """Calibrated Local score: ILS-family Top-k similarity weighted by word prob."""
    return state.w.reshape([1, 1]).cuda() * state.ils_top_k_cos_matrix


def context_consistency_score(state, bias):
    """Context-consistency score from the context-weighed difference and word prob."""
    return (bias - state.weighed_diff.mean().cpu()) * state.mean_context_prob


def score_mean_prob(state):
    """Mean probability of the word over its top-k context tokens."""
    return state.mean_context_prob


def score_svar(state):
    """SVAR score from the answer attention mass over the image region."""
    return state.svar.reshape([1, 1]).cpu()


# ---------------------------------------------------------------------------
# lm_head helpers (kept tiny to avoid recomputing logits per detector).
# ---------------------------------------------------------------------------

def _lm_head_probs(model, hidden, temperature):
    """Run lm_head on ``hidden`` and return softmax-over-vocab probabilities."""
    with torch.inference_mode():
        logits = model.model.lm_head(hidden).detach().cpu().float()
    return (temperature * logits).softmax(dim=-1)


def _lm_head_raw(model, hidden, temperature, image_start, image_end):
    """lm_head probabilities restricted to the image region, shaped for top-k."""
    with torch.inference_mode():
        logits = model.model.lm_head(hidden).detach().cpu().float()
    probs = (temperature * logits).softmax(dim=-1)           # [1, seq, vocab]
    probs = probs.unsqueeze(0)[:, :, image_start:image_end + 1]  # [1, 1, img_len, vocab]
    probs = probs.numpy().transpose(3, 0, 2, 1).max(axis=3)   # [vocab, 1, img_len]
    return probs


def _top_k_cos(token, detected_norm, image_embeddings, softmax_probs_raw, k, num_layers):
    """Top-k (Local) cosine similarity between a word and the image patches."""
    token_prob = softmax_probs_raw[token]
    top_k_indices = np.argsort(token_prob, axis=1)[:, -k:]
    top_k_embeddings = F.normalize(
        image_embeddings[torch.arange(num_layers).unsqueeze(1), top_k_indices, :],
        p=2, dim=-1,
    )
    cos_sim_k = torch.einsum(
        "id,jkd->ijk", detected_norm.unsqueeze(0), top_k_embeddings
    )
    return cos_sim_k.mean(dim=-1)


# ---------------------------------------------------------------------------
# Per-word shared computation.
# ---------------------------------------------------------------------------

def _build_word_state(
    *,
    token,
    softmax_probs_question,           # ILS instruction probs (w + context prob)
    glsim_softmax_probs_raw,
    ils_softmax_probs_raw,
    output_ids_rs,
    answer_hidden_states,
    glsim_image_embeddings,
    ils_image_embeddings,
    glsim_question_final_embeddings,
    ils_question_embeddings,
    glsim_image_layer,
    glsim_text_layer,
    ils_image_layer,
    ils_text_layer,
    answer_attention,
    image_start,
    image_end,
    layer_list,
    k,
    context_num,
):
    """Compute all shared per-word tensors and bundle them into a WordState.

    The GLSIM-family detectors (``global_cos_matrix``, ``top_k_cos_matrix``)
    use the ``glsim_*`` layers, while the ILS-family detectors
    (``calibrated_local``, ``context_consistency`` and the ILS combination)
    use the decoupled ``ils_*`` layers, so the two hyperparameter groups
    can be tuned independently.
    """
    w, _ = softmax_probs_question[:, token].max(dim=-1)
    toke_idx = torch.where(output_ids_rs == token)[0][0]

    detected_embeddings = answer_hidden_states[toke_idx - 1, :]
    detected_embeddings_norm = F.normalize(detected_embeddings, p=2, dim=-1)

    num_layers = glsim_image_embeddings.shape[0]

    # ---- GLSIM global ----
    global_emb_norm = F.normalize(glsim_question_final_embeddings, p=2, dim=-1)
    global_cos = torch.matmul(
        global_emb_norm, detected_embeddings_norm[glsim_image_layer].unsqueeze(0).T
    )
    global_cos, _ = global_cos.max(dim=0)

    # ---- GLSIM top_k (Local) ----
    glsim_top_k = _top_k_cos(
        token, detected_embeddings_norm[glsim_text_layer],
        glsim_image_embeddings, glsim_softmax_probs_raw, k, num_layers,
    )

    # ---- ILS top_k (Local), decoupled layers ----
    ils_top_k = _top_k_cos(
        token, detected_embeddings_norm[ils_text_layer],
        ils_image_embeddings, ils_softmax_probs_raw, k, num_layers,
    )

    # ---- Context block (ILS family: ils instruction + ils text) ----
    context_probs, context_index = torch.topk(
        softmax_probs_question[:, token], dim=0, k=context_num
    )
    context_embeddings = ils_question_embeddings[:, context_index, :].mean(dim=1)

    detected_embedding = detected_embeddings[ils_text_layer]
    diff_embeddings = (detected_embedding - context_embeddings).norm(dim=-1)
    detected_norm = detected_embedding.norm(dim=-1)
    weighed_diff = diff_embeddings / detected_norm

    mean_context_prob = context_probs.mean().reshape([1, 1]).cpu()

    # ---- SVAR (answer attention) ----
    if answer_attention == []:
        svar = torch.Tensor([0])
    else:
        svar = compute_svar(
            answer_attention[toke_idx - 1], image_start, image_end, layer_list
        )
        torch.cuda.synchronize()

    return WordState(
        token=token,
        w=w,
        detected_embeddings=detected_embeddings,
        detected_embeddings_norm=detected_embeddings_norm,
        global_cos_matrix=global_cos,
        top_k_cos_matrix=glsim_top_k,
        ils_top_k_cos_matrix=ils_top_k,
        mean_context_prob=mean_context_prob,
        weighed_diff=weighed_diff,
        svar=svar,
    )


# ---------------------------------------------------------------------------
# Top-level dispatcher (called by main in evaluate.py).
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_scores(
    args,
    vq_hidden_states,
    answer_hidden_states,
    output_ids,
    image_start,
    image_end,
    answer_start,
    tokens,
    final_ans,
    img_id,
    model,
    evaluator,
    answer_attention,
    k=32,
    # GLSIM-family layers (global_cos_matrix, top_k_cos_matrix).
    glsim_image_layer=None,
    glsim_text_layer=None,
    # Legacy flat names kept for backward compatibility -> map to GLSIM.
    image_layer=32,  # llava: 32, internvl: 28
    text_layer=31,  # llava: 31, internvl: 27
    instruction_layer=-1,
    # ILS-family layers (calibrated_local, context_consistency, ils).
    # Decoupled from GLSIM; default to the GLSIM layers when omitted.
    ils_image_layer=None,
    ils_text_layer=None,
    ils_instruction_layer=None,
    context_num=4,
    bias=4,  # llava: 1, internvl: 4
    layer_list=None,
    scale=0.1,
):
    """Compute GL_sim detection scores for one generated caption.

    Dispatches each detector (Global, Top-k, Calibrated-Local,
    Context-Consistency, Mean-Prob, SVAR) over every recalled and
    hallucinated word, and returns a dict with one list per score type, each
    containing the per-word scores for recalled (``*_true``) and hallucinated
    (``*_false``) objects.
    """
    if layer_list is None:
        layer_list = [i for i in range(4, 18)]

    # Resolve GLSIM / ILS layer groups. The ILS family is decoupled from
    # GLSIM: when ils_* are omitted they fall back to the GLSIM layers
    # (backward compatible with the old single-layer config).
    if glsim_image_layer is None:
        glsim_image_layer = image_layer
    if glsim_text_layer is None:
        glsim_text_layer = text_layer
    if ils_image_layer is None:
        ils_image_layer = glsim_image_layer
    if ils_text_layer is None:
        ils_text_layer = glsim_text_layer
    if ils_instruction_layer is None:
        ils_instruction_layer = instruction_layer

    temperature = scale
    # GLSIM-family embeddings / logits.
    glsim_image_embeddings = vq_hidden_states[glsim_image_layer, 0, image_start:image_end + 1].unsqueeze(0)
    glsim_question_final_embeddings = vq_hidden_states[glsim_text_layer, 0, -1, :].unsqueeze(0)
    glsim_softmax_probs_raw = _lm_head_raw(model, vq_hidden_states[glsim_image_layer], temperature, image_start, image_end)

    # ILS-family embeddings / logits (decoupled layers).
    ils_image_embeddings = vq_hidden_states[ils_image_layer, 0, image_start:image_end + 1].unsqueeze(0)
    ils_question_embeddings = vq_hidden_states[ils_instruction_layer, 0, image_end + 1:, :].unsqueeze(0)
    ils_softmax_probs_raw = _lm_head_raw(model, vq_hidden_states[ils_image_layer], temperature, image_start, image_end)
    ils_softmax_probs_question = _lm_head_probs(
        model, vq_hidden_states[ils_instruction_layer, 0, image_end + 1:], temperature
    )

    output_ids_rs = output_ids[answer_start:]
    cap_dict = evaluator.compute_hallucinations(img_id, final_ans, args)

    # One list per score type, split into recalled / hallucinated.
    matrices = {
        "global_cos_matrix_true": [], "global_cos_matrix_false": [],
        "top_k_cos_matrix_true": [], "top_k_cos_matrix_false": [],
        "calibrated_local_true": [], "calibrated_local_false": [],
        "context_consistency_true": [], "context_consistency_false": [],
        "mean_prob_matrix_true": [], "mean_prob_matrix_false": [],
        "svar_true": [], "svar_false": [],
    }

    with torch.no_grad():
        for target_indexes in _COLLECTED_INDEXES:
            collected_words = []
            for idx, true_idx in enumerate(cap_dict[target_indexes]):
                token, detected_word = _resolve_token(
                    args, cap_dict, target_indexes, idx, true_idx, tokens, output_ids_rs, model
                )
                if token is None:
                    continue

                # Recall words map to the "*_true" matrices, hallucinated to "*_false".
                suffix = "true" if target_indexes == "recall_idxs" else "false"

                if detected_word in collected_words:
                    continue
                collected_words.append(detected_word)

                # Skip words that never appeared in the generated answer.
                if torch.where(output_ids_rs == token)[0].numel() == 0:
                    continue

                state = _build_word_state(
                    token=token,
                    softmax_probs_question=ils_softmax_probs_question,
                    glsim_softmax_probs_raw=glsim_softmax_probs_raw,
                    ils_softmax_probs_raw=ils_softmax_probs_raw,
                    output_ids_rs=output_ids_rs,
                    answer_hidden_states=answer_hidden_states,
                    glsim_image_embeddings=glsim_image_embeddings,
                    ils_image_embeddings=ils_image_embeddings,
                    glsim_question_final_embeddings=glsim_question_final_embeddings,
                    ils_question_embeddings=ils_question_embeddings,
                    glsim_image_layer=glsim_image_layer,
                    glsim_text_layer=glsim_text_layer,
                    ils_image_layer=ils_image_layer,
                    ils_text_layer=ils_text_layer,
                    answer_attention=answer_attention,
                    image_start=image_start,
                    image_end=image_end,
                    layer_list=layer_list,
                    k=k,
                    context_num=context_num,
                )

                matrices[f"global_cos_matrix_{suffix}"].append(score_global(state))
                matrices[f"top_k_cos_matrix_{suffix}"].append(score_top_k(state))
                matrices[f"calibrated_local_{suffix}"].append(calibrated_local_score(state))
                matrices[f"context_consistency_{suffix}"].append(context_consistency_score(state, bias))
                matrices[f"mean_prob_matrix_{suffix}"].append(score_mean_prob(state))
                matrices[f"svar_{suffix}"].append(score_svar(state))

    return matrices
