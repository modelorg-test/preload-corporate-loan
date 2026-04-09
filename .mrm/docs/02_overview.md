# Credit Risk Model

## Overview
Binary classifier for retail credit risk assessment.

## Inputs
- `income`: Annual income (float)
- `debt_ratio`: Debt-to-income ratio (float)
- `credit_history_years`: Years of credit history (int)

## Outputs
- `risk_score`: Probability of default [0, 1]
- `risk_tier`: "low" | "medium" | "high"

## Performance
- AUC-ROC: 0.89
- Precision@10%FPR: 0.76
