# A/B Test Analysis: Designing and Reading an Experiment Correctly

A worked conversion experiment done the disciplined way, plus demonstrations of the two classic
ways teams manufacture false wins. The statistics live in a small, tested module; the notebook
uses it end to end.

## The experiment, and the decision

**Scenario:** a checkout redesign. Baseline conversion **12%**. We ship only if the new design
lifts conversion by at least **1.5 percentage points** (our minimum detectable effect and our
practical-significance bar).

**Design (before touching data):** to detect a +1.5pp lift with 80% power at the 5% level requires
**~7,760 users per arm**.

**Result:** treatment converted at **13.5%** vs control **11.9%**, an absolute lift of **+1.66pp**
(**+14% relative**). The 95% confidence interval on the lift is **[+0.6pp, +2.7pp]** (excludes
zero, p = 0.002).

**Decision: ship.** The effect is statistically real (CI excludes zero) and the point estimate
clears the +1.5pp practical bar, with the honest caveat that the lower CI bound sits just below the
bar, so a larger sample would tighten the estimate. A bare p-value would have said "significant"
and hidden that nuance.

## What this demonstrates

- **Design first.** Sample size is computed from the baseline, the effect worth detecting, alpha,
  and power, before any data is seen.
- **Report intervals and effect size, not a bare p-value.** Confidence interval on the lift,
  relative lift, and Cohen's h, so statistical and practical significance are separate questions.
- **An explicit ship decision** tied to a pre-registered threshold, not to whether p < 0.05.
- **A small, tested stats module** (`src/abtest/`) rather than a one-off script.

## The two pitfalls (shown on data with no real effect)

- **Peeking / optional stopping.** In an A/A test (no true difference), a single fixed-n test holds
  the false-positive rate near 5%, but "peek and stop when it looks significant" inflates it to
  **~20%**, and worse with more looks. Fix: fix the sample size in advance, or use sequential
  methods built for monitoring.
- **Multiple comparisons.** With `k` independent tests at 5%, the chance of at least one false
  positive is `1 - 0.95^k`: **40% at 10 metrics**, 64% at 20. Fix: correct the threshold
  (Bonferroni `alpha/k`, or FDR control).

## Run it

```bash
pip install -r requirements.txt
pip install -e .                      # makes `abtest` importable
pytest                                # 9 tests
jupyter notebook notebooks/analysis.ipynb
```

## Repository

```
ab-test-analysis/
├── README.md
├── requirements.txt
├── pyproject.toml
├── src/abtest/
│   ├── __init__.py
│   └── core.py           # sample size, two-proportion test + CI, effect size, pitfall simulators
├── tests/
│   └── test_core.py      # 9 tests
└── notebooks/
    └── analysis.ipynb    # design -> analyze -> decide -> pitfall demos
```

## Stack

Python, NumPy, SciPy, matplotlib; pytest for tests.
