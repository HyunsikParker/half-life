"""Choose the learning rate and loss balance on the VALIDATION learners.

The test learners are never touched here; evaluate.py scores them once, at the end.

    python sweep.py data/duolingo.csv.gz
"""

from __future__ import annotations

import sys

from hlr import HalfLifeRegression, features, observed_half_life, read_rows, split_of

GRID = [
    {"learning_rate": 0.01, "alpha": 0.001},
    {"learning_rate": 0.01, "alpha": 0.01},
    {"learning_rate": 0.01, "alpha": 0.1},
    {"learning_rate": 0.03, "alpha": 0.01},
]


def run(path, config):
    model = HalfLifeRegression(**config)
    for row in read_rows(path):
        if split_of(row["user_id"]) != "train":
            continue
        model.update(
            features(row["history_seen"], row["history_correct"], row["lexeme_id"]),
            row["days"],
            row["p"],
        )
    recall_error = half_life_error = 0.0
    counted = 0
    for row in read_rows(path):
        if split_of(row["user_id"]) != "validation":
            continue
        counted += 1
        p_hat, h_hat = model.predict(
            features(row["history_seen"], row["history_correct"], row["lexeme_id"]),
            row["days"],
        )
        recall_error += abs(p_hat - row["p"])
        half_life_error += abs(h_hat - observed_half_life(row["p"], row["days"]))
    return recall_error / counted, half_life_error / counted, model, counted


def main(path):
    best = None
    for config in GRID:
        mae, mae_h, model, counted = run(path, config)
        print(
            f"lr={config['learning_rate']:<6} alpha={config['alpha']:<6} "
            f"val MAE(p)={mae:.4f}  MAE(h)={mae_h:6.1f}d  n={counted:,}  "
            f"right {model.weights['right']:+.4f} wrong {model.weights['wrong']:+.4f} "
            f"bias {model.weights['bias']:+.4f}",
            flush=True,
        )
        if best is None or mae < best[0]:
            best = (mae, config)
    print(f"\nbest on validation: {best[1]}  MAE(p)={best[0]:.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/duolingo.csv.gz")
