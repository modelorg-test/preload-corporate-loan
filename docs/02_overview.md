# Overview & Strategy

Neural network model evaluating balance sheet health for institutional clients. Produces a Probability of Default (PD) and Loss Given Default (LGD) estimate mapped to an internal 15-grade rating scale.

## Inputs

- `ebitda_ratio`: EBITDA to total debt ratio (float)
- `leverage_ratio`: Total debt to total equity (float)
- `interest_coverage`: EBIT / interest expense (float)
- `sector_code`: GICS industry classification (categorical)

## Outputs

- `pd_estimate`: 12-month probability of default [0, 1]
- `lgd_estimate`: Expected loss given default [0, 1]
- `internal_grade`: Rating grade (1–15)
