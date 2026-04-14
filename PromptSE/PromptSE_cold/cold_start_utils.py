import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dsgat_style_drug_folds(num_drugs, n_folds, seed):
    rng = np.random.RandomState(seed)
    indices = np.arange(num_drugs)
    rng.shuffle(indices)
    fold_size = int(np.ceil(len(indices) / float(n_folds)))
    folds = []
    for fold_idx in range(n_folds):
        start = fold_idx * fold_size
        end = min(len(indices), start + fold_size)
        if start >= len(indices):
            break
        folds.append(indices[start:end].astype(int))
    return folds


def load_drug_index_file(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get(
                "drug_indices",
                payload.get(
                    "test_drugs",
                    payload.get("heldout_drug_indices", payload.get("heldout_drugs", [])),
                ),
            )
        return np.asarray(payload, dtype=int)
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=int)
    return np.asarray([int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()], dtype=int)


def save_split_manifest(path, folds):
    payload = {
        "fold_count": len(folds),
        "folds": [fold.astype(int).tolist() for fold in folds],
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def safe_roc_auc(y_true, y_score):
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_average_precision(y_true, y_score):
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def safe_pr_auc(y_true, y_score):
    if np.unique(y_true).size < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return float(auc(recall, precision))


def safe_curve_auc(y_true, y_score):
    if np.unique(y_true).size < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(auc(fpr, tpr))


def sigmoid_array(scores):
    scores = np.asarray(scores, dtype=np.float32)
    scores = np.clip(scores, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-scores))


def compute_binary_classification_metrics(
    score_matrix,
    label_matrix,
    score_threshold=0.5,
    label_threshold=0.5,
    apply_sigmoid=False,
):
    labels = np.asarray(label_matrix, dtype=np.float32)
    scores = np.asarray(score_matrix, dtype=np.float32)
    if apply_sigmoid:
        scores = sigmoid_array(scores)

    binary_labels = (labels >= label_threshold).astype(int)
    flat_labels = binary_labels.reshape(-1)
    flat_scores = scores.reshape(-1)
    pred_labels = (flat_scores > score_threshold).astype(int)

    if np.unique(flat_labels).size < 2:
        auc_value = float("nan")
        aupr_value = float("nan")
        mcc_value = float("nan")
        f1_value = float("nan")
        macro_f1_value = float("nan")
    else:
        auc_value = safe_curve_auc(flat_labels, flat_scores)
        aupr_value = safe_pr_auc(flat_labels, flat_scores)
        mcc_value = float(matthews_corrcoef(flat_labels, pred_labels))
        f1_value = float(f1_score(flat_labels, pred_labels, zero_division=0))
        macro_f1_value = float(f1_score(flat_labels, pred_labels, average="macro", zero_division=0))

    return {
        "AUC": auc_value,
        "AUPR": aupr_value,
        "F1_SCORE": f1_value,
        "MACRO_F1": macro_f1_value,
        "MCC": mcc_value,
        "score_threshold": float(score_threshold),
        "label_threshold": float(label_threshold),
    }


def compute_positive_regression_metrics(score_matrix, label_matrix):
    positive_mask = label_matrix != 0
    y_true = label_matrix[positive_mask].astype(float)
    y_score = score_matrix[positive_mask].astype(float)
    if y_true.size < 2 or np.unique(y_true).size < 2:
        return {
            "pearson": float("nan"),
            "rMSE": float("nan"),
            "spearman": float("nan"),
            "MAE": float("nan"),
            "regression_supported": False,
        }

    pearson = float(np.corrcoef(y_true, y_score)[0, 1])
    spearman = float(stats.spearmanr(y_true, y_score)[0])
    rmse = float(np.sqrt(np.mean((y_true - y_score) ** 2)))
    mae = float(mean_absolute_error(y_true, y_score))
    return {
        "pearson": pearson,
        "rMSE": rmse,
        "spearman": spearman,
        "MAE": mae,
        "regression_supported": True,
    }


def precision_at(actual, predicted, cutoff):
    overlap = set(actual) & set(predicted[:cutoff])
    return float(len(overlap)) / float(cutoff)


def recall_at(actual, predicted, cutoff):
    overlap = set(actual) & set(predicted[:cutoff])
    return float(len(overlap)) / float(len(set(actual)))


def ndcg_at(actual, ranked_items, cutoff):
    dcg = 0.0
    relevance = []
    for idx in range(cutoff):
        if ranked_items[idx] in actual:
            dcg += 1 / math.log(idx + 2, 2)
            relevance.append(1)
        else:
            relevance.append(0)
    relevance.sort(reverse=True)
    idcg = sum(relevance[idx] / math.log(idx + 2, 2) for idx in range(cutoff))
    return 0.0 if idcg == 0 else dcg / idcg


def map_auc(pos_indices, neg_indices, scores):
    map_value = 0.0
    auc_count = 0.0
    pos_scores = scores[pos_indices]
    neg_scores = scores[neg_indices]
    pos_scores = pos_scores[np.argsort(pos_scores)[::-1]]
    neg_scores = neg_scores[np.argsort(neg_scores)[::-1]]
    for rank, pos_score in enumerate(pos_scores):
        larger_neg = 0.0
        for neg_score in neg_scores:
            if pos_score <= neg_score:
                larger_neg += 1
            else:
                auc_count += 1
        map_value += (rank + 1) / (rank + larger_neg + 1)
    return map_value / len(pos_indices), auc_count / (len(pos_indices) * len(neg_indices))


def evaluate_others(score_matrix, positions):
    precision_values = np.zeros(len(positions))
    recall_values = np.zeros(len(positions))
    map_value = 0.0
    ndcg_value = 0.0
    auc_value = 0.0
    binary_matrix = (score_matrix["labels"] != 0).astype(int)
    labels = score_matrix["labels"]
    scores = score_matrix["scores"]

    for row_idx in range(labels.shape[0]):
        positive_indices = np.flatnonzero(binary_matrix[row_idx]).tolist()
        if not positive_indices:
            continue
        all_indices = np.arange(labels.shape[1])
        ranked = all_indices[np.argsort(scores[row_idx, all_indices])[::-1]]
        precision_values += np.array([precision_at(positive_indices, ranked, pos) for pos in positions])
        recall_values += np.array([recall_at(positive_indices, ranked, pos) for pos in positions])
        ndcg_value += ndcg_at(positive_indices, ranked, min(10, len(ranked)))
        negative_indices = np.setdiff1d(all_indices, positive_indices, assume_unique=True)
        map_row, auc_row = map_auc(np.asarray(positive_indices), negative_indices, scores[row_idx])
        map_value += map_row
        auc_value += auc_row

    row_count = labels.shape[0]
    return (
        map_value / row_count,
        auc_value / row_count,
        ndcg_value / row_count,
        precision_values / row_count,
        recall_values / row_count,
    )


def compute_dsgat_style_metrics(score_matrix, label_matrix, positions=(1, 5, 10, 15)):
    labels = np.asarray(label_matrix)
    scores = np.asarray(score_matrix)
    binary_labels = (labels != 0).astype(int)

    single_drug_auc = []
    single_drug_aupr = []
    for row_idx in range(labels.shape[0]):
        single_drug_auc.append(safe_roc_auc(binary_labels[row_idx], scores[row_idx]))
        single_drug_aupr.append(safe_average_precision(binary_labels[row_idx], scores[row_idx]))

    all_auc = safe_roc_auc(binary_labels.reshape(-1), scores.reshape(-1))
    all_aupr = safe_average_precision(binary_labels.reshape(-1), scores.reshape(-1))
    map_value, _, ndcg, prec, rec = evaluate_others({"scores": scores, "labels": labels}, list(positions))

    metrics = compute_positive_regression_metrics(scores, labels)
    metrics.update(
        {
            "auc_all": all_auc,
            "aupr_all": all_aupr,
            "drugAUC": float(np.nanmean(np.asarray(single_drug_auc, dtype=float))),
            "drugAUPR": float(np.nanmean(np.asarray(single_drug_aupr, dtype=float))),
            "MAP": float(map_value),
            "nDCG": float(ndcg),
            "P1": float(prec[0]),
            "P5": float(prec[1]),
            "P10": float(prec[2]),
            "P15": float(prec[3]),
            "R1": float(rec[0]),
            "R5": float(rec[1]),
            "R10": float(rec[2]),
            "R15": float(rec[3]),
        }
    )
    return metrics
