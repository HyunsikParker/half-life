"""Fit half-life regression and compare it with the schedulers people actually use.

    python evaluate.py data/duolingo.csv.gz

Writes docs/results.json (the numbers) and docs/model.json (weights small enough
to run the prediction in a browser, so the demo needs no server).

Learners are split by user id, not by review: every review from one person lands
entirely in train or entirely in test.  A review-level split would let the model
see the same person's other reviews of the same word and would flatter every
model here, this one included.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from hlr import (
    HalfLifeRegression,
    Leitner,
    Pimsleur,
    clamp,
    features,
    observed_half_life,
    read_rows,
    split_of,
)

LEXEMES_IN_DEMO = 400
EVAL_SPLIT = "test"


def auc(labels, scores):
    """Rank-based AUC; ties share the average rank."""
    pairs = sorted(zip(scores, labels))
    ranks, index = [0.0] * len(pairs), 0
    while index < len(pairs):
        stop = index
        while stop + 1 < len(pairs) and pairs[stop + 1][0] == pairs[index][0]:
            stop += 1
        average = (index + stop) / 2.0 + 1
        for position in range(index, stop + 1):
            ranks[position] = average
        index = stop + 1
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    positive_rank_sum = sum(r for r, (_, label) in zip(ranks, pairs) if label)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def main(path, eval_split="test"):
    global EVAL_SPLIT
    EVAL_SPLIT = eval_split
    model = HalfLifeRegression()
    leitner, pimsleur = Leitner(), Pimsleur()

    lexeme_counts = Counter()
    lexeme_names = {}
    train_rows = test_rows = 0
    recall_sum = 0.0

    print("pass 1/2 — training on the training users", flush=True)
    for row in read_rows(path):
        lexeme_counts[row["lexeme_id"]] += 1
        lexeme_names.setdefault(row["lexeme_id"], row["lexeme_string"])
        if split_of(row["user_id"]) != "train":
            test_rows += 1
            continue
        train_rows += 1
        recall_sum += row["p"]
        model.update(
            features(row["history_seen"], row["history_correct"], row["lexeme_id"]),
            row["days"],
            row["p"],
        )
        if train_rows % 2_000_000 == 0:
            print(f"  {train_rows:,} training reviews", flush=True)

    mean_recall = recall_sum / max(1, train_rows)
    print(f"  trained on {train_rows:,} reviews; {test_rows:,} held out", flush=True)

    print(f"pass 2/2 — scoring the {EVAL_SPLIT} learners", flush=True)
    errors = defaultdict(float)
    half_life_errors = defaultdict(float)
    labels, scores = [], defaultdict(list)
    counted = 0

    for row in read_rows(path):
        if split_of(row["user_id"]) != EVAL_SPLIT:
            continue
        counted += 1
        days, p = row["days"], row["p"]
        truth = observed_half_life(p, days)
        x = features(row["history_seen"], row["history_correct"], row["lexeme_id"])

        predictions = {
            "hlr": model.predict(x, days),
            "leitner": None,
            "pimsleur": None,
            "constant": (mean_recall, float("nan")),
        }
        for name, scheduler in (("leitner", leitner), ("pimsleur", pimsleur)):
            h = scheduler.half_life(row["history_seen"], row["history_correct"])
            predictions[name] = (clamp(2.0 ** (-days / h), 0.0001, 1.0), h)

        for name, (p_hat, h_hat) in predictions.items():
            errors[name] += abs(p_hat - p)
            if not math.isnan(h_hat):
                half_life_errors[name] += abs(h_hat - truth)
            scores[name].append(p_hat)
        labels.append(1 if p >= 1.0 else 0)

    results = {
        "train_reviews": train_rows,
        "test_reviews": counted,
        "split": "by learner, 80/10/10 train/validation/test (no learner spans two sets)",
        "eval_split": EVAL_SPLIT,
        "mean_recall": mean_recall,
        "models": {},
    }
    for name in ("hlr", "leitner", "pimsleur", "constant"):
        results["models"][name] = {
            "mae_recall": errors[name] / counted,
            "mae_half_life_days": (half_life_errors[name] / counted) if name != "constant" else None,
            "auc": auc(labels, scores[name]),
        }
        row = results["models"][name]
        half_life = row["mae_half_life_days"]
        print(
            f"  {name:9s} MAE(p)={row['mae_recall']:.4f}  AUC={row['auc']:.4f}"
            + (f"  MAE(half-life)={half_life:.1f}d" if half_life is not None else "")
        )

    Path("docs").mkdir(exist_ok=True)
    Path("docs/results.json").write_text(json.dumps(results, indent=1))

    demo_lexemes = [
        {
            "id": lexeme_id,
            "text": lexeme_names[lexeme_id].split("/")[0],
            "tag": lexeme_names[lexeme_id],
            "weight": model.weights[f"lex:{lexeme_id}"],
            "reviews": count,
        }
        for lexeme_id, count in lexeme_counts.most_common(LEXEMES_IN_DEMO)
    ]
    Path("docs/model.json").write_text(
        json.dumps(
            {
                "right": model.weights["right"],
                "wrong": model.weights["wrong"],
                "bias": model.weights["bias"],
                "lexemes": demo_lexemes,
            },
            indent=1,
        )
    )
    print("wrote docs/results.json and docs/model.json")


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else "data/duolingo.csv.gz",
        sys.argv[2] if len(sys.argv) > 2 else "test",
    )
