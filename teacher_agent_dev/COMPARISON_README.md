# Strategy Comparison: Teacher vs Baselines

## Overview

This module compares three training strategies for the student agent:

1. **Random Strategy**: Student receives random questions from task generator until they can confidently pass difficult questions
2. **Progressive Strategy**: Student receives questions in progressive difficulty order (Easy → Medium → Hard) within each family sequentially
3. **Teacher Strategy**: RL teacher agent learns optimal curriculum using UCB bandit algorithm

## Goal

Demonstrate that the **Teacher-trained student performs best** - achieving highest accuracy on difficult questions.

## Running the Comparison

```bash
cd teacher_agent_dev
python compare_strategies.py
```

This will:
- Train all three strategies for 500 iterations
- Track accuracy on general questions and difficult questions
- Generate comparison plots showing all three strategies
- Print summary statistics

## Output

### Plot: `comparison_all_strategies.png`

The plot contains three subplots:

1. **General Accuracy Over Time**: Shows how student accuracy improves on medium-difficulty questions
2. **Difficult Question Accuracy**: **KEY METRIC** - Shows accuracy on hard questions (most important for demonstrating teacher superiority)
3. **Learning Efficiency**: Bar chart showing iterations to reach 75% target vs final performance

### Key Metrics Tracked

- **General Accuracy**: Student performance on medium-difficulty questions from all topics
- **Difficult Accuracy**: Student performance on hard-difficulty questions (target metric)
- **Iterations to Target**: How many iterations until student reaches 75% accuracy on difficult questions
- **Final Accuracy**: Final performance after 500 iterations

## Expected Results

The Teacher strategy should show:
- ✅ **Highest final accuracy** on difficult questions
- ✅ **Efficient learning** (good balance of speed and performance)
- ✅ **Better curriculum** (smarter topic/difficulty selection)

### Example Output

```
STRATEGY COMPARISON SUMMARY
======================================================================
Random          | ✅ Reached       | Iterations:   51 | Final Acc: 0.760
Progressive     | ✅ Reached       | Iterations:  310 | Final Acc: 0.520
Teacher         | ✅ Reached       | Iterations:   55 | Final Acc: 0.880
======================================================================
```

**Teacher wins with highest final accuracy!**

## Strategy Details

### Random Strategy
- Completely random selection of topics and difficulties
- No curriculum structure
- Baseline for comparison
- May reach target quickly due to luck, but doesn't optimize learning

### Progressive Strategy
- Rigid curriculum: Easy → Medium → Hard for each topic sequentially
- No adaptation to student needs
- Slow to reach difficult questions
- Doesn't account for forgetting or optimal pacing

### Teacher Strategy
- **RL-based curriculum learning**
- Uses UCB bandit to balance exploration/exploitation
- Adapts based on student improvement (reward signal)
- Optimizes for efficient learning
- Can strategically review topics to prevent forgetting

## Visualization Features

- **Color coding**: Teacher in green (highlighted as best), Random in red, Progressive in teal
- **Line styles**: Teacher with solid thick line, baselines with dashed/dotted
- **Annotations**: Final accuracy values labeled on plots
- **Target line**: 75% accuracy threshold marked on difficult question plot
- **Summary statistics**: Table showing which strategies reached target and when

## Customization

You can modify parameters in `compare_strategies.py`:

```python
num_iterations = 500  # Number of training iterations
target_accuracy = 0.75  # Target accuracy on difficult questions
seed = 42  # Random seed for reproducibility
```

## Files

- `compare_strategies.py` - Main comparison script
- `comparison_all_strategies.png` - Generated comparison plot
- `train_teacher.py` - Teacher training logic
- `mock_student.py` - Student agent implementation
- `mock_task_generator.py` - Task generator

## Notes

- All strategies use the same student parameters for fair comparison
- Evaluation uses held-out test sets
- Teacher strategy learns from rewards based on student improvement
- Results may vary slightly due to randomness, but teacher should consistently outperform baselines

