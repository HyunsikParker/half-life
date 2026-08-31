"""Half-life regression for spaced repetition, plus the schedulers it is compared against.

A learner's memory of one item decays roughly exponentially.  Write the time at
which recall has fallen to 50% as the item's *half-life* h.  Then after a gap of
d days the probability of recalling it is

    p = 2 ** (-d / h)

Half-life regression (Settles & Meeder, ACL 2016) makes h a function of what the
learner has done with that item:

    h = 2 ** (theta . x)

where x counts previous successes and failures and identifies the item itself.
Fitting theta to real review logs gives a per-item, per-learner half-life instead
of the one-size-fits-all interval a fixed scheduler uses.

Trained by SGD on the loss from the paper:

    (p_hat - p) ** 2  +  alpha * (h_hat - h) ** 2  +  lambda * ||theta|| ** 2

The second term is supervised by the half-life implied by the observed recall,
h = -d / log2(p), which is only defined for 0 < p < 1 and is clamped.
"""

from __future__ import annotations

import csv
import gzip
import math
from collections import defaultdict

SECONDS_PER_DAY = 86400.0
MIN_HALF_LIFE = 15.0 / (24 * 60)  # 15 minutes, in days
MAX_HALF_LIFE = 274.0             # 9 months, in days
CLAMP_P = 0.0001


def clamp(value, low, high):
    return max(low, min(high, value))


def observed_half_life(p, days):
    """Half-life implied by recalling with probability p after `days` days."""
    p = clamp(p, CLAMP_P, 1 - CLAMP_P)
    return clamp(-days / math.log(p, 2), MIN_HALF_LIFE, MAX_HALF_LIFE)


def features(history_seen, history_correct, lexeme_id):
    """Feature vector as (name, value) pairs.

    Square roots are used for the counts, as in the paper: the tenth correct
    answer should move the estimate less than the second one did.
    """
    wrong = max(0, history_seen - history_correct)
    return [
        ("right", math.sqrt(1 + history_correct)),
        ("wrong", math.sqrt(1 + wrong)),
        ("bias", 1.0),
        (f"lex:{lexeme_id}", 1.0),
    ]


class HalfLifeRegression:
    def __init__(self, learning_rate=0.01, alpha=0.001, l2=0.1):
        self.weights = defaultdict(float)
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.l2 = l2
        self.seen = defaultdict(int)

    def half_life(self, x):
        dot = sum(self.weights[name] * value for name, value in x)
        return clamp(2.0 ** clamp(dot, -20.0, 20.0), MIN_HALF_LIFE, MAX_HALF_LIFE)

    def predict(self, x, days):
        h = self.half_life(x)
        return clamp(2.0 ** (-days / h), CLAMP_P, 1.0), h

    def update(self, x, days, p):
        p_hat, h_hat = self.predict(x, days)
        h = observed_half_life(p, days)
        # d/dtheta of (p_hat - p)^2 and alpha * (h_hat - h)^2, both through
        # h = 2^(theta.x), so each carries a factor of h * ln 2.
        ln2 = math.log(2.0)
        d_recall = 2.0 * (p_hat - p) * (ln2 ** 2) * p_hat * days / h_hat
        d_half = 2.0 * (h_hat - h) * self.alpha * ln2 * h_hat
        for name, value in x:
            self.seen[name] += 1
            rate = self.learning_rate / math.sqrt(self.seen[name])
            gradient = (d_recall + d_half) * value
            self.weights[name] -= rate * gradient
            self.weights[name] -= rate * self.l2 * self.weights[name] / math.sqrt(self.seen[name])


class Leitner:
    """Classic box scheduler: each success doubles the interval, each miss halves it."""

    name = "Leitner"

    def half_life(self, history_seen, history_correct, _lexeme_id=None):
        wrong = max(0, history_seen - history_correct)
        return clamp(2.0 ** clamp(history_correct - wrong, -20.0, 20.0), MIN_HALF_LIFE, MAX_HALF_LIFE)


class Pimsleur:
    """Fixed exponential schedule driven only by how many times an item was seen."""

    name = "Pimsleur"

    def half_life(self, history_seen, _history_correct=None, _lexeme_id=None):
        exponent = clamp(2.35 * history_seen - 16.46, -20.0, 20.0)
        return clamp(2.0 ** exponent, MIN_HALF_LIFE, MAX_HALF_LIFE)


def split_of(user_id):
    """train / validation / test, decided by learner so no person spans two sets.

    Hyper-parameters are chosen on the validation learners; the test learners are
    scored once, at the end.
    """
    digest = 0
    for char in user_id:
        digest = (digest * 131 + ord(char)) % 1_000_003
    bucket = digest % 100
    if bucket < 80:
        return "train"
    return "validation" if bucket < 90 else "test"


def read_rows(path, limit=None):
    """Stream the learning-trace file, yielding one dict per review."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if limit is not None and index >= limit:
                return
            days = float(row["delta"]) / SECONDS_PER_DAY
            if days <= 0:
                continue
            yield {
                "p": clamp(float(row["p_recall"]), CLAMP_P, 1.0),
                "days": days,
                "user_id": row["user_id"],
                "lexeme_id": row["lexeme_id"],
                "lexeme_string": row["lexeme_string"],
                "history_seen": int(row["history_seen"]),
                "history_correct": int(row["history_correct"]),
            }
