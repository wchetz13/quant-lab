# Purged cross-validation for financial machine learning

Most "profitable" ML trading models found in research do not survive live trading. A leading
cause is **leakage in cross-validation**. Standard k-fold assumes samples are independent.
Financial samples are not:

- **Overlapping labels.** A label like "sign of the return over the next 20 days" depends on
  days $t+1, \dots, t+20$. Its neighbour's label shares 19 of those 20 days.
- **Overlapping features.** Rolling-window features (trailing returns, rolling volatility) are
  near-duplicates for adjacent days.

Shuffled k-fold scatters those near-duplicates across the train/test split. A flexible model,
such as a random forest, then predicts a test sample by recognising its neighbour in the
training set and copying that neighbour's label. The result is measured skill that cannot exist
out of sample.

## What's here

`purged_cv.py`:

- **`PurgedKFold`** is a scikit-learn splitter (López de Prado, 2018, ch. 7). Given each label's
  time window $[s_i, e_i]$, it uses contiguous test folds and:
  - **purges** every training sample whose label window overlaps the test fold's label span;
  - **embargoes** a span of training samples after each test fold, since their trailing-window
    features still contain test-period prices.
- **`walk_forward_splits`** is an expanding-window baseline. It trains only on labels that are
  fully resolved before the test period begins.
- **`make_dataset`** produces daily returns, trailing-window features at 1 to 60 days plus
  rolling vol, and 20-day forward-sign labels. With `signal=0` the returns are i.i.d., so the
  **true AUC of any model is exactly 0.5**. With `signal > 0`, a trailing-mean feature shifts
  future drift and plants real predictability.

## Results

The model is a random forest (200 trees, minimum leaf size 5). Each run uses 3,000 samples and
5 folds. Scores are cross-validated **ROC AUC**, averaged over 5 seeds (± 1 s.e.). AUC is used
because the labels are imbalanced, so accuracy has no clean chance level.

| CV scheme | Pure noise (true AUC = 0.5) | Planted signal |
|---|---|---|
| Shuffled k-fold | **0.699 ± 0.008** | **0.810 ± 0.007** |
| Contiguous k-fold | 0.522 ± 0.016 | 0.670 ± 0.012 |
| Purged k-fold | 0.519 ± 0.017 | 0.670 ± 0.012 |
| Purged k-fold + 60-day embargo | 0.519 ± 0.015 | 0.667 ± 0.014 |
| Walk-forward | 0.520 ± 0.017 | 0.655 ± 0.013 |

- **Shuffled k-fold:** on data with no signal at all, it reports AUC 0.70, which would pass for
  a strong trading signal. Where a real signal exists, it inflates the model's skill by about
  0.14 AUC.
- **Time-aware schemes:** all four agree with each other and with the truth to within noise.
  They recover chance on noise and about 0.67 on the planted signal.

**Purging vs. contiguous folds.** In this setup, most of the leak comes from shuffling. With
contiguous folds, only samples within a label horizon of each fold boundary can leak: about
2 × 20 of each 600-sample fold. Purging removes exactly those, and the embargo removes the
feature overlap. So purging is a **guarantee**, checked property-style in the tests, rather than
a big number in this table. Its effect grows as the label horizon gets long relative to fold
size, and it matters most for combinatorial and nested CV, where many fold boundaries cut
through the training data.

## Tests

`test_purged_cv.py` checks:

- **Purge guarantee:** over random sample counts, horizons, fold counts and embargoes, no
  training label window ever intersects the test fold's label span.
- **Splitter mechanics:** test folds partition the samples, the embargo removes exactly the
  intended indices, and walk-forward trains only on resolved past labels.
- **Labels:** each label is the sign of the sum of the next $h$ one-day returns, rebuilt
  independently from the features.
- **Leakage:** shuffled k-fold finds skill in pure noise (AUC > 0.62), while purged CV reports
  chance.
- **Positive control:** purged CV still detects a planted signal.

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley, ch. 7.
- Bailey, D., Borwein, J., López de Prado, M., Zhu, Q. (2014). Pseudo-mathematics and financial charlatanism: the effects of backtest overfitting on out-of-sample performance. *Notices of the AMS* 61(5).
