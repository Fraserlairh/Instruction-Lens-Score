import numpy as np
import sklearn as sk
from sklearn.metrics import roc_auc_score, average_precision_score
def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = roc_auc_score(labels, examples)
    aupr = average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr

from sklearn.metrics import f1_score, accuracy_score
def find_best_threshold(true_scores, false_scores, metric='f1'):
    """
    找到最佳阈值，使得 F1 或 Accuracy 最大。
    
    Args:
        true_scores (list or array): 正样本的置信度
        false_scores (list or array): 负样本的置信度
        metric (str): 'f1' 或 'acc'
    
    Returns:
        best_threshold, best_f1, best_acc
    """
    true_scores = np.array(true_scores)
    false_scores = np.array(false_scores)

    # 合并所有分数和对应标签
    all_scores = np.concatenate([true_scores, false_scores])
    labels = np.concatenate([np.ones_like(true_scores), np.zeros_like(false_scores)])

    # 候选阈值：所有 unique 分数（也可以加边界）
    thresholds = np.unique(all_scores)
    # 可选：加入 min-1 和 max+1 保证覆盖极端情况
    thresholds = np.concatenate([[thresholds.min() - 1], thresholds, [thresholds.max() + 1]])

    best_score      = -1
    best_threshold  = None
    best_f1         = None
    best_acc        = None

    for th in thresholds:
        preds = (all_scores >= th).astype(int)
        f1 = f1_score(labels, preds)
        acc = accuracy_score(labels, preds)

        if metric == 'f1':
            current_score = f1
        else:
            current_score = acc

        if current_score > best_score:
            best_score = current_score
            best_threshold = th
            best_f1 = f1
            best_acc = acc

    return best_threshold, best_f1, best_acc