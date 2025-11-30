# Randomness Update: Configurable Seeds & Variance Analysis

## Issue

Previously, `compare_strategies.py` always used `seed=42`, making results **identical every run**. This:
- ✅ Good for reproducibility
- ❌ Hides the stochastic nature of learning
- ❌ Doesn't show variance in results
- ❌ Makes it hard to assess robustness

## Solution

Added command-line arguments for configurable randomness:

### Usage Options

**1. Random seed (default - results vary each run):**
```bash
python compare_strategies.py
# Uses current time as seed - different results each run
```

**2. Deterministic (reproducible - same results every time):**
```bash
python compare_strategies.py --deterministic
# Uses seed=42 - identical results for reproducibility
```

**3. Specific seed:**
```bash
python compare_strategies.py --seed 123
# Uses seed=123 - reproducible but different from default
```

**4. Variance analysis (multiple runs):**
```bash
python compare_strategies.py --runs 10
# Runs 10 times with different seeds, shows mean ± std
```

**5. Custom iterations:**
```bash
python compare_strategies.py --iterations 1000
# Train for 1000 iterations instead of default 500
```

### Example: Variance Analysis

```bash
python compare_strategies.py --runs 5 --iterations 200
```

Output:
```
VARIANCE ANALYSIS ACROSS RUNS
======================================================================

Random:
  Final Accuracy: 0.653 ± 0.042 (range: 0.600 - 0.707)
  Iterations to Target: 378.2 ± 45.3 (range: 320 - 445)

Progressive:
  Final Accuracy: 0.360 ± 0.028 (range: 0.330 - 0.390)
  Iterations to Target: 499.0 ± 0.0 (range: 499 - 499)

Teacher:
  Final Accuracy: 0.773 ± 0.035 (range: 0.720 - 0.813)
  Iterations to Target: 258.4 ± 32.1 (range: 210 - 305)
```

This shows:
- **Mean performance** across runs
- **Standard deviation** (variance)
- **Range** (min-max)

## Why This Matters

1. **Shows stochasticity**: Random and Teacher strategies have natural variance
2. **Assesses robustness**: Large variance = less reliable
3. **Realistic expectations**: Single-run results may be lucky/unlucky
4. **Better comparisons**: Variance analysis shows if differences are significant

## Default Behavior Change

- **Before**: Always `seed=42` (deterministic)
- **After**: Default uses current time (random, varies each run)
- **To get old behavior**: Use `--deterministic` flag

## Best Practices

- **Development/Debugging**: Use `--deterministic` for consistent testing
- **Final Evaluation**: Use `--runs 10` or more for robust statistics
- **Quick Tests**: Default (random) is fine for seeing variance
- **Reproducing Results**: Use `--seed <number>` to reproduce specific runs

## Implementation Details

- All strategies use the same seed for fair comparison
- Variance analysis computes mean, std, and range across runs
- Plots show first run (or can be modified to show averaged curves)
- Seed is printed so runs can be reproduced

