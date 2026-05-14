# Methodology & Assumptions

A feedforward neural network (3 hidden layers, 64-32-16 units) trained on historical SEC filings and internal cash flow ratios. Dropout regularisation (0.3) applied to prevent overfitting on the relatively small corporate dataset.

The PD estimate is calibrated to the bank's long-run average default rate using Platt scaling.
