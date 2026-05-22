# Estimation techniques for software projects

## T-shirt sizing
XS / S / M / L / XL. Used early in planning when precision is impossible. Maps roughly to story points: XS=1, S=2, M=5, L=8, XL=13.

## Planning poker
Team members independently pick a Fibonacci-like number (1, 2, 3, 5, 8, 13, 21). Discuss outliers, re-vote until consensus.

## PERT (Three-point estimation)
For each task estimate Optimistic (O), Most Likely (M), Pessimistic (P). Weighted average: `(O + 4M + P) / 6`. Captures uncertainty.

## Reference-class forecasting
Find similar past projects, scale their duration to the current scope. Avoids overconfidence.

## Buffer rules of thumb
- Add 15-20% for medium-confidence estimates.
- Add 30-50% for high-uncertainty work (research, integrations with external systems).
- App Store/Google Play review: 3-7 days additional.
