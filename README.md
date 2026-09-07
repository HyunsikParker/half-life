# Half-Life

**When will you forget this?** Flashcard apps answer with a fixed rule — get it right, wait twice
as long. This fits the decay to real review logs instead, and schedules from that.

Trained on 10,243,562 reviews and scored on 1,279,602 from learners held out entirely, drawn from
the 12.9-million-review Duolingo learning-traces release.

Live demo: https://hyunsikparker.github.io/half-life/

## The result

Scored on 1,279,602 reviews from learners held out entirely — no person's reviews appear in both
training and test, so nothing here is scored on someone the model has already met.

| scheduler | mean abs. error on recall | AUC | half-life error |
|---|---|---|---|
| **Half-life regression (learned)** | **0.1208** | 0.5405 | 116.3 d |
| Always predict the average | 0.1727 | 0.5000 | — |
| Leitner (fixed doubling) | 0.2224 | 0.5496 | 147.4 d |
| Pimsleur (fixed exponential) | 0.4367 | 0.5275 | 154.3 d |

The learned model predicts recall with **46% less error than Leitner** and 30% less than simply
guessing the average. It is also closest on the half-life itself, by about a month.

Leitner has higher AUC (0.550 vs 0.541), which measures ranking. The learned model has lower
mean absolute error on recall probabilities. MAE is not a separate calibration test, and these
offline metrics do not establish whether following either schedule improves learning.

## How it works

Memory of one item decays roughly exponentially. Write the time at which recall has fallen to 50%
as that item's **half-life** `h`. After a gap of `d` days,

```
p = 2 ** (-d / h)
```

Half-life regression (Settles & Meeder, ACL 2016) makes `h` a function of the learner's history
with that specific item:

```
h = 2 ** (theta . x)      x = [ sqrt(1 + times recalled),
                                sqrt(1 + times forgotten),
                                bias,
                                which item this is ]
```

`theta` is fit by SGD on `(p_hat - p)^2 + alpha * (h_hat - h)^2 + lambda * ||theta||^2`, where the
second term is supervised by the half-life implied by the observed recall, `h = -d / log2(p)`.

A fixed scheduler has no term for *which item this is*. Across the 400 most-reviewed items the
learned item weights span a 1.7x range in half-life — a real difference that Leitner cannot
express, because it gives a Spanish irregular and the word "a" the same interval.

## What the weights do and do not tell you

Fitted on the full data: `times recalled` −0.035, `times forgotten` −0.105, bias +7.54.

The sign on `times recalled` is negative, which would read as "recalling something makes you forget
it faster". **Do not read it that way.** This is observational data: Duolingo itself chose when to
test each learner, and it waits longer before re-testing items you have seen many times. So the
exposure counts partly absorb the app's own scheduling policy rather than a property of memory. The
effect is small (about 9% across the whole slider range) next to `times forgotten`, whose negative
sign is both larger and the direction memory research would predict.

Getting a causal read on those coefficients needs randomised review intervals, which this dataset
does not contain. The predictive claim above does not depend on them.

## Reproducing

```
pip install -r requirements.txt        # no third-party packages are actually needed
python sweep.py     data/duolingo.csv.gz   # picks hyper-parameters on the validation learners
python evaluate.py  data/duolingo.csv.gz   # scores the test learners, writes docs/
```

`evaluate.py` downloads the dataset on first run (379 MB), then writes `docs/results.json` and
`docs/model.json`. Pure Python and the standard library — no numpy, no GPU, no API key, about six
minutes on a laptop CPU.

Learners are split 80/10/10 by user id. Hyper-parameters were chosen on the validation learners;
the test learners were scored once, at the end.

## The demo

`docs/` is a static page that runs the whole prediction in the browser from `model.json` — no
server, no network call after load. Pick something you are learning, say how it has gone so far,
and it draws the decay curve, marks when Leitner would review you, and marks when the model would.
The page labels those values as model predictions and shows the observational-data limitation.
Controls stay disabled until both data files load; failed loads show an error without a prediction.

## Data

Duolingo learning traces, 12,854,226 reviews — Settles & Meeder, *A Trainable Spaced Repetition
Model for Language Learning*, ACL 2016. <https://doi.org/10.7910/DVN/N8XJME>, CC BY 4.0.
Downloaded at build time; not redistributed here.

## Licence

MIT, see `LICENSE`.
