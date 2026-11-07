# MNIST error analysis

Derived from the full 10,000-example official test-set confusion matrix in [the recorded run](../reports/mnist-test.json). Predictions were made after checkpoint selection by validation loss. These are descriptive results for one seed.

| Digit | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| 0 | 980 | 97.70% | 99.49% | 98.58% |
| 1 | 1135 | 99.38% | 99.47% | 99.43% |
| 2 | 1032 | 97.98% | 98.74% | 98.36% |
| 3 | 1010 | 99.11% | 98.71% | 98.91% |
| 4 | 982 | 97.79% | 98.98% | 98.38% |
| 5 | 892 | 99.21% | 98.32% | 98.76% |
| 6 | 958 | 99.78% | 96.66% | 98.20% |
| 7 | 1028 | 98.15% | 98.25% | 98.20% |
| 8 | 974 | 97.36% | 98.36% | 97.85% |
| 9 | 1009 | 98.60% | 97.82% | 98.21% |

## Most frequent confusion pairs

- True 7 predicted as 2: 13 examples.
- True 6 predicted as 4: 12 examples.
- True 6 predicted as 0: 9 examples.
- True 9 predicted as 4: 6 examples.
- True 8 predicted as 0: 6 examples.

There are 150 errors in total. Digit 6 has the lowest recall in this run. The aggregate matrix identifies where to inspect images; it does not establish why an individual image was misclassified. Changes motivated by these test errors need a new held-out evaluation rather than repeatedly tuning against the same test set.
