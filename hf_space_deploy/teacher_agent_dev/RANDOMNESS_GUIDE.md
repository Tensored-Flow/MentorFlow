# Randomness Configuration Guide

## Quick Answer to Your Question

**Yes, it's fine to have randomness!** By default, the script now uses **random seeds**, so results will vary each run. This is actually **better** because it shows the true stochastic nature of learning.

## How It Works Now

### Default Behavior (Random - Results Vary)
```bash
python compare_strategies.py
```
- Uses current time as seed
- **Results will be different each run**
- Better for seeing variance and stochasticity

### Deterministic Mode (Same Results Every Time)
```bash
python compare_strategies.py --deterministic
```
- Uses fixed seed=42
- **Results will be identical every run**
- Good for debugging and reproducibility

### Variance Analysis (Multiple Runs)
```bash
python compare_strategies.py --runs 10
```
- Runs 10 times with different seeds
- Shows mean ± standard deviation
- Best for robust evaluation

## Why This Matters

The learning process has natural randomness:
- **Random strategy**: Obviously random! 🎲
- **Student learning**: Stochastic answers (probabilistic)
- **Teacher strategy**: RL exploration adds variance

Seeing this variance is important because:
1. **Single runs can be lucky/unlucky**
2. **Variance shows robustness** (lower variance = more reliable)
3. **Real-world performance will vary**

## Example: Seeing the Difference

**Run 1:**
```
Teacher: Final Acc: 0.773
Random:  Final Acc: 0.653
```

**Run 2 (different seed):**
```
Teacher: Final Acc: 0.789
Random:  Final Acc: 0.641
```

**Run 3 (different seed):**
```
Teacher: Final Acc: 0.761
Random:  Final Acc: 0.667
```

This variance is **normal and expected**! The teacher should still outperform on average.

## Best Practices

1. **For development/testing**: Use `--deterministic` for consistent debugging
2. **For evaluation**: Use `--runs 10` to see robust statistics
3. **For quick checks**: Default (random) is fine - just run multiple times manually

## All Options

```bash
python compare_strategies.py [OPTIONS]

Options:
  --seed SEED          Use specific seed (e.g., --seed 123)
  --deterministic      Use seed=42 (reproducible, same every time)
  --iterations N       Train for N iterations (default: 500)
  --runs N             Run N times for variance analysis
```

## Summary

✅ **Default now has randomness** - results vary (this is good!)
✅ **Use --deterministic** if you want identical results
✅ **Use --runs N** for proper variance analysis
✅ **Variance is expected** - shows realistic behavior

The stochastic nature is actually a feature, not a bug! It shows the true variability in learning.

