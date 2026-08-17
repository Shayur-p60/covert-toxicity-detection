# Covert Toxicity Detection

Comparing classical ML (TF-IDF + Logistic Regression / SVM) against a
fine-tuned BERT model on a 3-class toxicity task, with a specific focus
on toxic comments that **don't** rely on obvious profanity or slurs —
the cases where keyword-based moderation tends to fail.

**Authors:** Shayur Pragdheen & Tharish Dayanand

## The problem

Most toxicity filters lean heavily on keyword/profanity matching. That
catches explicit abuse but misses toxic language that's harmful without
using any flagged words — condescension, backhanded insults, coded
hostility. This project asks: how much better does a fine-tuned
transformer do on that harder case, compared to classical bag-of-words
models?

## Approach

Comments from the [Jigsaw Toxic Comment Classification](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge)
dataset (~160K Wikipedia talk-page comments) are relabeled into 3
classes:

| Class | Definition |
|---|---|
| **Clean** | Not toxic per the original Jigsaw labels |
| **Explicit Toxic** | Toxic *and* matches a profanity/slur/insult keyword list |
| **Covert Toxic** | Toxic *but* doesn't match that keyword list |

Three models are trained and compared on an identical, stratified
70/15/15 train/validation/test split:

1. **Logistic Regression** (TF-IDF features, class-balanced weighting)
2. **Linear SVM** (same features/weighting)
3. **BERT** (`bert-base-uncased`, fine-tuned with a class-weighted loss)

All three are evaluated the same way — macro F1 and per-class F1 — and
model selection during BERT training deliberately optimizes for
**covert-class F1**, not overall accuracy, since accuracy on this
dataset is dominated by the large "Clean" majority class and would
reward a model for basically ignoring the class we actually care about.

## Results

Test set (held out, never touched during training or hyperparameter selection):

| Metric | Logistic Regression | Linear SVM | BERT |
|---|---|---|---|
| Accuracy | 0.902 | 0.946 | **0.961** |
| Macro F1 | 0.733 | 0.780 | **0.835** |
| Clean F1 | 0.945 | 0.971 | **0.979** |
| Explicit Toxic F1 | 0.865 | 0.878 | **0.886** |
| **Covert Toxic F1** | 0.387 | 0.493 | **0.640** |

BERT's advantage widens substantially on the harder class: a 65%
relative improvement in Covert Toxic F1 over Logistic Regression, and
30% over the (already class-weighted) SVM baseline. That gap — not the
overall accuracy numbers, which are close and somewhat inflated by the
easy majority class — is the actual result this project is about.

Full per-split, per-class breakdowns (train/validation/test) are in
[`results/`](results/).

## Limitations — read before citing these numbers

- **"Covert toxic" is a keyword-heuristic label, not human-annotated
  subtlety.** It's operationalized here as *toxic per the original
  Jigsaw label, but absent from a profanity/slur/insult regex list* —
  not an independent human judgment of "this is toxic in a subtle
  way." That's a reasonable, transparent simplification for a course
  project, but it means the task is closer to "detect toxicity that
  doesn't rely on flagged keywords" than a deeper notion of
  rhetorical subtlety. A stronger version of this project would use
  human-annotated covert/implicit toxicity labels (e.g. from
  implicit hate speech datasets) rather than a keyword proxy.
- **Single train/val/test split, single random seed.** Results aren't
  averaged over multiple splits or seeds, so treat the exact numbers
  as indicative rather than statistically airtight.
- The keyword list itself is hand-built (see the notebook) and not
  independently validated — it's a reasonable first pass, not a
  vetted lexicon.

## Running this yourself

```bash
pip install -r requirements.txt
```

Get the dataset — see [`data/README.md`](data/README.md) for the
Kaggle download steps (not included in this repo; it's a few hundred
MB and freely available from Kaggle directly).

Then open [`notebooks/covert_toxicity_detection.ipynb`](notebooks/covert_toxicity_detection.ipynb)
and run it top to bottom. It expects `data/train.csv` to exist (from
the step above) and will:
1. Explore the data and build the 3-class labels (Phase 1)
2. Train and evaluate the TF-IDF + Logistic Regression / SVM baselines (Phase 2)
3. Fine-tune and evaluate BERT (Phase 3) — a GPU is strongly
   recommended for this part; on CPU it will be very slow

All intermediate artifacts (processed CSVs, trained models, split
data) are regenerated locally and aren't committed to the repo — see
`.gitignore`.

## Project structure

```
covert-toxicity-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── covert_toxicity_detection.ipynb
├── data/
│   └── README.md          # how to get the dataset (not committed)
└── results/
    ├── model_comparison_test.csv
    └── model_comparison_validation.csv
```
