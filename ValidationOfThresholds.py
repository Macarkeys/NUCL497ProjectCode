import pandas as pd
import numpy as np

# Load test predictions
df = pd.read_csv('results/test_predictions.csv')

# test thresholds from 0.01 to 0.99
thresholds = np.linspace(0.01, 0.99, 99)
best_threshold = 0.5
best_f1 = -1

for t in thresholds:
  # Convert probabilities to 1 or 0 based on the threshold t
  preds = (df['p_stable'] >= t).astype(int)

  # Calculate True Positives, False Positives, False Negatives
  tp = np.sum((preds == 1) & (df['is_stable'] == 1))
  fp = np.sum((preds == 1) & (df['is_stable'] == 0))
  fn = np.sum((preds == 0) & (df['is_stable'] == 1))

  # Calculate F1 score
  precision = tp / (tp + fp) if (tp + fp) > 0 else 0
  recall = tp / (tp + fn) if (tp + fn) > 0 else 0
  f1 = (
      (2 * precision * recall) / (precision + recall)
      if (precision + recall) > 0
      else 0
  )

  if f1 > best_f1:
    best_f1 = f1
    best_threshold = t

print(f'Best Threshold: {best_threshold:.2f} with F1-score: {best_f1:.3f}')