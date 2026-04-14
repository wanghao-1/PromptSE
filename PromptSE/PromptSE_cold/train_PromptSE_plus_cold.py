from __future__ import division
from __future__ import print_function

import argparse
import csv
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from cold_start_utils import (
    build_dsgat_style_drug_folds,
    compute_binary_classification_metrics,
    sigmoid_array,
    load_drug_index_file,
    save_split_manifest,
    set_random_seed,
)
from model_PromptSE_plus import DSEModel
from utils import load_data, resolve_project_path, save_all


SCRIPT_DIR = Path(__file__).resolve().parent


def build_timestamp():
    return time.strftime("%Y%m%d_%H%M%S")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run DSGAT-style drug cold-start evaluation with the MLDSP+ model."
    )
    parser.add_argument("--data_path", type=str, default=str(resolve_project_path(default_subdir="data")))
    parser.add_argument("--params_path", type=str, default=str(resolve_project_path(default_subdir="params")))
    parser.add_argument("--result_path", type=str, default=str(resolve_project_path(default_subdir="result_cold")))
    parser.add_argument("--log_dir", type=str, default=str(resolve_project_path(default_subdir="log_cold")))
    parser.add_argument("--matrix_file", type=str, default="drug_se_matrix.txt")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--split_mode", choices=["drug", "custom_drug"], default="drug")
    parser.add_argument("--custom_test_drugs", type=str, default="")
    parser.add_argument("--n_folds", type=int, default=10)
    parser.add_argument("--fold_limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--hid_dim", type=int, default=64)
    parser.add_argument("--fpt_dim", type=int, default=128)
    parser.add_argument("--vec_dim", type=int, default=300)
    parser.add_argument("--gnn_dim", type=int, default=617)
    parser.add_argument("--vec_len", type=int, default=100)
    parser.add_argument("--kge_dim", type=int, default=400)
    parser.add_argument("--bio_dim", type=int, default=768)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--kk", type=int, default=5)
    parser.add_argument("--pos_weight", type=float, default=40.0)
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--label_threshold", type=float, default=0.5)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--D_n", type=int, default=1020)
    parser.add_argument("--S_n", type=int, default=5599)
    parser.add_argument("--save_model", action="store_true", default=False)
    parser.add_argument("--checkpoint_subdir", type=str, default="checkpoints")
    parser.add_argument("--no_pretrain", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    return parser


def get_device(force_cpu):
    use_cuda = torch.cuda.is_available() and not force_cpu
    return torch.device("cuda:0" if use_cuda else "cpu")


def configure_logging(log_dir, run_name, timestamp):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_name}_{timestamp}.log"
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    return log_path


def build_run_name(args):
    if args.run_name:
        return args.run_name
    split_tag = "custom" if args.split_mode == "custom_drug" else f"{args.n_folds}fold"
    return (
        f"mldspp_dsgat_like_{split_tag}_seed{args.seed}_"
        f"ep{args.epochs}_lr{args.lr}_wd{args.weight_decay}_kk{args.kk}"
    )


def create_run_dir(base_dir, run_name):
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    run_dir = base_dir / run_name
    if run_dir.exists():
        run_dir = base_dir / f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def move_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {key: move_to_cpu(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [move_to_cpu(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_cpu(value) for value in obj)
    return obj


def save_fold_checkpoint(model, optimizer, args, fold_idx, test_drugs, run_dir, metrics):
    checkpoint_dir = Path(run_dir) / args.checkpoint_subdir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"model_fold{fold_idx:02d}.pt"
    payload = {
        "fold": fold_idx,
        "epoch": args.epochs,
        "run_name": Path(run_dir).name,
        "split_mode": args.split_mode,
        "test_drug_indices": np.asarray(test_drugs, dtype=np.int64),
        "metrics": move_to_cpu(metrics),
        "args": vars(args).copy(),
        "model_state_dict": move_to_cpu(model.state_dict()),
        "optimizer_state_dict": move_to_cpu(optimizer.state_dict()),
    }
    torch.save(payload, checkpoint_path)
    logging.info("Saved fold %02d checkpoint to %s", fold_idx, checkpoint_path)
    return checkpoint_path


def get_edge_index_and_weights(adj_matrix, device):
    row, col = np.nonzero(adj_matrix)
    edge_index = torch.tensor(np.vstack((row, col)), dtype=torch.long, device=device)
    edge_weight = torch.tensor(adj_matrix[row, col], dtype=torch.float32, device=device)
    return edge_index, edge_weight


def process_matrix(matrix, group_count):
    row_nonzero = np.count_nonzero(matrix, axis=1)
    sorted_indices = np.argsort(row_nonzero)[::-1]
    matrix = matrix[sorted_indices][:, sorted_indices]

    num = matrix.shape[0]
    lower_triangular_ones = np.tril(np.ones((num, num), dtype=np.float32))
    block_mask = np.zeros((num, num), dtype=np.float32)
    group_size_ratio = 1 / group_count
    start_index = 0
    while start_index < num:
        end_index = min(num, start_index + math.ceil(group_size_ratio * num))
        block_mask[start_index:end_index, start_index:end_index] = 1
        start_index = end_index

    value_matrix = lower_triangular_ones + block_mask
    value_matrix[value_matrix != 0] = 1
    matrix = matrix * value_matrix

    reverse_indices = np.argsort(sorted_indices)
    matrix = matrix[reverse_indices][:, reverse_indices]
    return matrix.T


def symmetric_normalize(matrix):
    degree = matrix.sum(axis=1)
    inv_sqrt = np.zeros_like(degree, dtype=np.float32)
    nonzero = degree > 0
    inv_sqrt[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    return (inv_sqrt[:, None] * matrix * inv_sqrt[None, :]).astype(np.float32)


def build_graphs(train_matrix, args, device):
    drug_graph = symmetric_normalize(train_matrix @ train_matrix.T)
    se_graph = symmetric_normalize(train_matrix.T @ train_matrix)
    se_graph = process_matrix(se_graph, args.kk)
    edge_index_to_drug_drug, weight_drug = get_edge_index_and_weights(drug_graph, device)
    edge_index_to_se_se, weight_se = get_edge_index_and_weights(se_graph, device)
    return edge_index_to_drug_drug, edge_index_to_se_se, weight_se, weight_drug


def prepare_features(args, device):
    return load_data(
        path=args.data_path,
        params_path=args.params_path,
        num=args.D_n,
        map_location=device,
    )


def load_matrix(args):
    matrix_path = Path(args.data_path) / args.matrix_file
    matrix = np.loadtxt(matrix_path).astype(np.float32)
    if matrix.shape != (args.D_n, args.S_n):
        raise ValueError(
            f"Matrix shape mismatch: expected {(args.D_n, args.S_n)}, got {matrix.shape} from {matrix_path}"
        )
    return matrix


def build_folds(args, num_drugs):
    if args.split_mode == "custom_drug":
        if not args.custom_test_drugs:
            raise ValueError("--custom_test_drugs is required when --split_mode custom_drug is used.")
        return [load_drug_index_file(args.custom_test_drugs)]
    return build_dsgat_style_drug_folds(num_drugs, args.n_folds, args.seed)


def create_model(args, device, feature_bundle, graph_bundle):
    (
        fpt_feature,
        mpnn_feature,
        weave_feature,
        afp_feature,
        nf_feature,
        vec_feature,
        llm_feature,
        w2v_feature,
        checkpoint,
    ) = feature_bundle
    edge_index_to_drug_drug, edge_index_to_se_se, weight_se, weight_drug = graph_bundle

    fpt_feature = fpt_feature.to(device)
    mpnn_feature = mpnn_feature.to(device)
    weave_feature = weave_feature.to(device)
    afp_feature = afp_feature.to(device)
    nf_feature = nf_feature.to(device)
    vec_feature = vec_feature.to(device)

    model = DSEModel(
        args,
        fpt_feature,
        mpnn_feature,
        weave_feature,
        afp_feature,
        nf_feature,
        vec_feature,
        edge_index_to_drug_drug,
        edge_index_to_se_se,
        weight_se,
        weight_drug,
        llm_feature,
        w2v_feature,
    ).to(device)

    if not args.no_pretrain and isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        logging.info("Loaded checkpoint with %d missing keys and %d unexpected keys.", len(missing_keys), len(unexpected_keys))

    param_group_attn_alpha = [model.attn_alpha]
    param_group_rest = [p for p in model.parameters() if p is not model.attn_alpha]
    optimizer = optim.Adam(
        [
            {"params": param_group_rest, "lr": args.lr},
            {"params": param_group_attn_alpha, "lr": 0.0001},
        ],
        weight_decay=args.weight_decay,
    )
    return model, optimizer


def train_one_epoch(model, optimizer, train_mask, target_tensor, pos_weight):
    model.train()
    optimizer.zero_grad()
    outputs, _, _ = model()
    loss = F.binary_cross_entropy_with_logits(
        outputs[train_mask],
        target_tensor[train_mask],
        pos_weight=pos_weight,
    )
    loss.backward()
    optimizer.step()
    return loss.item(), outputs.detach()


def run_fold(args, fold_idx, test_drugs, matrix, feature_bundle, device, run_dir):
    train_mask_np = np.ones((args.D_n, args.S_n), dtype=bool)
    train_mask_np[test_drugs, :] = False
    target_tensor = torch.from_numpy(matrix).to(device)
    train_mask = torch.from_numpy(train_mask_np).to(device)
    test_mask = torch.from_numpy(~train_mask_np).to(device)
    train_matrix = matrix.copy()
    train_matrix[test_drugs, :] = 0

    graph_bundle = build_graphs(train_matrix, args, device)
    model, optimizer = create_model(args, device, feature_bundle, graph_bundle)
    pos_weight = torch.tensor([args.pos_weight], dtype=torch.float32, device=device)

    fold_start = time.time()
    outputs = None
    for epoch in range(args.epochs):
        loss_value, outputs = train_one_epoch(model, optimizer, train_mask, target_tensor, pos_weight)
        if (epoch + 1) % args.log_interval == 0 or epoch == 0 or epoch + 1 == args.epochs:
            logging.info(
                "fold=%02d epoch=%04d loss=%.6f train_drugs=%d test_drugs=%d",
                fold_idx,
                epoch + 1,
                loss_value,
                args.D_n - len(test_drugs),
                len(test_drugs),
            )

    # Recompute logits once in inference mode so BatchNorm/dropout-dependent
    # modules follow evaluation behavior during testing.
    model.eval()
    with torch.no_grad():
        outputs, _, _ = model()

    fold_logits = outputs[test_drugs].detach().cpu().numpy()
    fold_scores = sigmoid_array(fold_logits)
    fold_labels = matrix[test_drugs]
    metrics = compute_binary_classification_metrics(
        fold_scores,
        fold_labels,
        score_threshold=args.score_threshold,
        label_threshold=args.label_threshold,
        apply_sigmoid=False,
    )
    metrics["fold"] = fold_idx
    metrics["train_drugs"] = int(args.D_n - len(test_drugs))
    metrics["test_drugs"] = int(len(test_drugs))
    metrics["duration_sec"] = round(time.time() - fold_start, 3)

    if args.save_model:
        save_fold_checkpoint(model, optimizer, args, fold_idx, test_drugs, run_dir, metrics)

    save_all(outputs, test_mask, fold_idx, path=str(run_dir))
    return metrics, fold_scores, fold_labels


def write_fold_metrics(result_csv, rows):
    fieldnames = [
        "fold",
        "train_drugs",
        "test_drugs",
        "duration_sec",
        "AUC",
        "AUPR",
        "F1_SCORE",
        "MACRO_F1",
        "MCC",
        "score_threshold",
        "label_threshold",
    ]
    with open(result_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarise_metrics(rows):
    numeric_keys = ["AUC", "AUPR", "F1_SCORE", "MACRO_F1", "MCC", "score_threshold", "label_threshold"]
    summary = {}
    for key in numeric_keys:
        values = np.asarray([row[key] for row in rows], dtype=float)
        if np.isnan(values).all():
            summary[key] = None
        else:
            summary[key] = float(np.nanmean(values))
    return summary


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.run_name = build_run_name(args)
    run_timestamp = build_timestamp()
    run_dir = create_run_dir(args.result_path, args.run_name)
    log_path = configure_logging(args.log_dir, run_dir.name, run_timestamp)
    set_random_seed(args.seed)
    device = get_device(args.cpu)
    logging.info("Arguments: %s", args)
    logging.info("Device: %s", device)
    logging.info("Run timestamp: %s", run_timestamp)

    matrix = load_matrix(args)
    folds = build_folds(args, args.D_n)
    if args.fold_limit > 0:
        folds = folds[: args.fold_limit]
    save_split_manifest(run_dir / "split_manifest.json", folds)

    feature_bundle = prepare_features(args, device)

    result_rows = []
    all_scores = []
    all_labels = []
    for fold_idx, test_drugs in enumerate(folds, start=1):
        logging.info("Starting fold %02d with %d held-out drugs.", fold_idx, len(test_drugs))
        metrics, fold_scores, fold_labels = run_fold(args, fold_idx, test_drugs, matrix, feature_bundle, device, run_dir)
        result_rows.append(metrics)
        all_scores.append(fold_scores)
        all_labels.append(fold_labels)
        logging.info(
            "Fold %02d summary: AUC=%.4f AUPR=%.4f F1=%.4f MacroF1=%.4f MCC=%.4f",
            fold_idx,
            metrics["AUC"],
            metrics["AUPR"],
            metrics["F1_SCORE"],
            metrics["MACRO_F1"],
            metrics["MCC"],
        )

    result_csv = run_dir / "MLDSPP_cold_result.csv"
    write_fold_metrics(result_csv, result_rows)
    np.savetxt(run_dir / "blind_pred.csv", np.vstack(all_scores), delimiter=",")
    np.savetxt(run_dir / "blind_raw.csv", np.vstack(all_labels), delimiter=",")

    summary = summarise_metrics(result_rows)
    summary_payload = {
        "run_name": run_dir.name,
        "device": str(device),
        "log_path": str(log_path),
        "fold_count": len(result_rows),
        "split_mode": args.split_mode,
        "seed": args.seed,
        "save_model": args.save_model,
        "checkpoint_dir": str(run_dir / args.checkpoint_subdir) if args.save_model else None,
        "mean_metrics": summary,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    logging.info("Average metrics: %s", summary)
    logging.info("Outputs saved to %s", run_dir)


if __name__ == "__main__":
    main()
