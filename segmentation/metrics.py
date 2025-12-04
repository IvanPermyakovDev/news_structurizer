from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report, precision_recall_fscore_support


def compute_boundary_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor = None
) -> Dict[str, float]:
    """
    Compute metrics for boundary detection task.

    Returns:
        - precision: for boundary class (1)
        - recall: for boundary class (1)
        - f1: for boundary class (1)
        - accuracy: overall token accuracy
        - pk: Pk metric (segmentation penalty)
        - window_diff: WindowDiff metric
    """
    pred_flat = []
    label_flat = []

    for i in range(predictions.shape[0]):
        for j in range(predictions.shape[1]):
            if labels[i, j] != -100:
                pred_flat.append(predictions[i, j].item())
                label_flat.append(labels[i, j].item())

    if len(pred_flat) == 0:
        return {'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0}

    pred_arr = np.array(pred_flat)
    label_arr = np.array(label_flat)

    precision, recall, f1, _ = precision_recall_fscore_support(
        label_arr, pred_arr, labels=[1], average='binary', zero_division=0
    )

    accuracy = (pred_arr == label_arr).mean()

    metrics = {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy),
    }

    pk, wd = compute_segmentation_metrics(predictions, labels)
    metrics['pk'] = pk
    metrics['window_diff'] = wd

    return metrics


def compute_segmentation_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    k: int = None
) -> Tuple[float, float]:
    """
    Compute Pk and WindowDiff metrics.
    Lower is better for both metrics.
    """
    pk_scores = []
    wd_scores = []

    for i in range(predictions.shape[0]):
        pred_boundaries = []
        true_boundaries = []

        for j in range(predictions.shape[1]):
            if labels[i, j] != -100:
                pred_boundaries.append(predictions[i, j].item())
                true_boundaries.append(labels[i, j].item())

        if len(pred_boundaries) == 0:
            continue

        if k is None:
            num_true_segs = sum(true_boundaries) + 1
            k = max(1, len(true_boundaries) // (2 * num_true_segs)) if num_true_segs > 0 else 1
            k = min(k, len(true_boundaries) - 1) if len(true_boundaries) > 1 else 1

        pk = compute_pk(pred_boundaries, true_boundaries, k)
        wd = compute_window_diff(pred_boundaries, true_boundaries, k)

        pk_scores.append(pk)
        wd_scores.append(wd)

    if not pk_scores:
        return 0.0, 0.0

    return float(np.mean(pk_scores)), float(np.mean(wd_scores))


def compute_pk(pred: List[int], ref: List[int], k: int) -> float:
    """Compute Pk metric."""
    n = len(ref)
    if n <= k or k <= 0:
        return 0.0

    errors = 0
    total = 0

    pred_seg = boundaries_to_segments(pred)
    ref_seg = boundaries_to_segments(ref)

    for i in range(n - k):
        same_pred = pred_seg[i] == pred_seg[i + k]
        same_ref = ref_seg[i] == ref_seg[i + k]

        if same_pred != same_ref:
            errors += 1
        total += 1

    return errors / total if total > 0 else 0.0


def compute_window_diff(pred: List[int], ref: List[int], k: int) -> float:
    """Compute WindowDiff metric."""
    n = len(ref)
    if n <= k or k <= 0:
        return 0.0

    errors = 0
    total = 0

    for i in range(n - k):
        pred_boundaries = sum(pred[i:i + k])
        ref_boundaries = sum(ref[i:i + k])

        if pred_boundaries != ref_boundaries:
            errors += 1
        total += 1

    return errors / total if total > 0 else 0.0


def boundaries_to_segments(boundaries: List[int]) -> List[int]:
    """Convert boundary labels to segment IDs."""
    segments = []
    current_seg = 0

    for b in boundaries:
        if b == 1 and segments:
            current_seg += 1
        segments.append(current_seg)

    return segments


def get_classification_report(
    predictions: torch.Tensor,
    labels: torch.Tensor
) -> str:
    """Get sklearn classification report."""
    pred_flat = []
    label_flat = []

    for i in range(predictions.shape[0]):
        for j in range(predictions.shape[1]):
            if labels[i, j] != -100:
                pred_flat.append(predictions[i, j].item())
                label_flat.append(labels[i, j].item())

    return classification_report(
        label_flat, pred_flat,
        target_names=['continuation', 'boundary'],
        zero_division=0
    )


def _match_with_tolerance(pred: Sequence[int], true: Sequence[int], tolerance: int) -> Tuple[int, int, int]:
    matched_true: set[int] = set()
    tp = 0
    for p in pred:
        best_idx = None
        best_dist = tolerance + 1
        for idx, t in enumerate(true):
            if idx in matched_true:
                continue
            dist = abs(p - t)
            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx is not None:
            matched_true.add(best_idx)
            tp += 1
    fp = len(pred) - tp
    fn = len(true) - tp
    return tp, fp, fn


def boundary_f1(
    pred_boundaries: Sequence[Sequence[int]],
    true_boundaries: Sequence[Sequence[int]],
    *,
    tolerance: int = 1,
) -> Tuple[float, float, float]:
    tp_total = fp_total = fn_total = 0
    for pred, true in zip(pred_boundaries, true_boundaries):
        tp, fp, fn = _match_with_tolerance(pred, true, tolerance)
        tp_total += tp
        fp_total += fp
        fn_total += fn

    precision = tp_total / (tp_total + fp_total) if tp_total + fp_total else 0.0
    recall = tp_total / (tp_total + fn_total) if tp_total + fn_total else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def _boundary_vector(boundaries: Sequence[int], total_units: int) -> List[int]:
    vec = [0] * max(total_units - 1, 0)
    for b in boundaries:
        if 0 <= b < len(vec):
            vec[b] = 1
    return vec


def window_diff(true: Sequence[int], pred: Sequence[int], total_units: int) -> float:
    if total_units <= 1:
        return 0.0
    avg_seg_len = max(1, round(total_units / (len(true) + 1)))
    comparisons = total_units - avg_seg_len
    if comparisons <= 0:
        return 0.0

    ref_vec = _boundary_vector(true, total_units)
    pred_vec = _boundary_vector(pred, total_units)

    errors = 0
    for start in range(comparisons):
        end = start + avg_seg_len - 1
        ref_sum = sum(ref_vec[start:end])
        pred_sum = sum(pred_vec[start:end])
        if ref_sum != pred_sum:
            errors += 1
    return errors / comparisons


def _segment_ids(boundaries: Sequence[int], total_units: int) -> List[int]:
    ids: List[int] = []
    current = 0
    boundary_set = set(boundaries)
    for idx in range(total_units):
        ids.append(current)
        if idx in boundary_set:
            current += 1
    return ids


def pk_metric(true: Sequence[int], pred: Sequence[int], total_units: int) -> float:
    if total_units <= 1:
        return 0.0
    avg_seg_len = max(1, round(total_units / (len(true) + 1)))
    comparisons = total_units - avg_seg_len
    if comparisons <= 0:
        return 0.0

    true_ids = _segment_ids(true, total_units)
    pred_ids = _segment_ids(pred, total_units)

    errors = 0
    for start in range(comparisons):
        same_true = true_ids[start] == true_ids[start + avg_seg_len]
        same_pred = pred_ids[start] == pred_ids[start + avg_seg_len]
        if same_true != same_pred:
            errors += 1
    return errors / comparisons


@dataclass
class MetricAccumulator:
    threshold: float = 0.5
    tolerance: int = 1
    preds: List[List[int]] = field(default_factory=list)
    trues: List[List[int]] = field(default_factory=list)
    lengths: List[int] = field(default_factory=list)

    def update(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> None:
        probs = torch.sigmoid(logits).detach().cpu()
        labels = labels.detach().cpu()
        mask = mask.detach().cpu().bool()

        for prob_vec, label_vec, mask_vec in zip(probs, labels, mask):
            valid_len = mask_vec.sum().item()
            if valid_len == 0:
                continue
            valid_probs = prob_vec[:valid_len]
            valid_labels = label_vec[:valid_len]
            pred_idx = [i for i, score in enumerate(valid_probs.tolist()) if score >= self.threshold]
            true_idx = [i for i, value in enumerate(valid_labels.tolist()) if value >= 0.5]
            self.preds.append(pred_idx)
            self.trues.append(true_idx)
            self.lengths.append(valid_len)

    def compute(self) -> Dict[str, float]:
        if not self.preds:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "window_diff": 0.0, "pk": 0.0}

        precision, recall, f1 = boundary_f1(self.preds, self.trues, tolerance=self.tolerance)
        wd_scores = [window_diff(t, p, l) for t, p, l in zip(self.trues, self.preds, self.lengths)]
        pk_scores = [pk_metric(t, p, l) for t, p, l in zip(self.trues, self.preds, self.lengths)]
        window_diff_avg = sum(wd_scores) / len(wd_scores) if wd_scores else 0.0
        pk_avg = sum(pk_scores) / len(pk_scores) if pk_scores else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "window_diff": window_diff_avg,
            "pk": pk_avg,
        }
