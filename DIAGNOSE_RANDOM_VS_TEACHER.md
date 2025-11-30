# Diagnosing Random Strategy Outperforming Teacher

## Problem

Random strategy is performing better than Teacher strategy. You suspect this is due to task generator incoherence.

## Analysis Needed

Let me investigate:

1. **Task Generator Coherence**: Are tasks independent or do they build on each other?
2. **Teacher Reward Function**: Is it properly rewarding curriculum coherence?
3. **Student Learning Model**: Does it actually benefit from coherent curriculum?
4. **Teacher Exploration**: Is teacher exploring too much instead of exploiting?

---

## Investigation Plan

Checking the actual behavior to diagnose the issue.

