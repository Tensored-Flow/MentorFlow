# Teacher Agent Development System

A complete teacher agent system for developing and testing meta-RL curriculum learning algorithms independently.

## Overview

This system provides:
- **Mock Student Agent**: Realistic student with learning + forgetting (Ebbinghaus curve)
- **Mock Task Generator**: Simple task generator with multiple topics and difficulties
- **Teacher Agent**: UCB (Upper Confidence Bound) bandit algorithm for curriculum sequencing
- **Training Loop**: Complete training system with evaluation
- **Visualization**: Plotting utilities for analysis

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Run Tests

```bash
python test_teacher.py
```

This verifies:
- Student learns with practice
- Student forgets over time
- Teacher explores actions
- Teacher exploits good actions

### 2. Train Teacher Agent

```bash
python train_teacher.py
```

Expected output:
```
======================================================================
TEACHER AGENT TRAINING
======================================================================
Iterations: 500
Evaluation tasks: 15
Action space: 30 actions
======================================================================
Iteration   0 | Student Acc: 0.267 | Avg Reward: 0.850 | Action: his-ea-N
Iteration  50 | Student Acc: 0.453 | Avg Reward: 1.120 | Action: sci-me-R
...
Iteration 500 | Student Acc: 0.812 | Avg Reward: 0.780 | Action: lit-ha-N
```

### 3. Generate Visualizations

```python
from train_teacher import train_teacher
from visualize import *

# Train teacher
history, teacher, student = train_teacher(num_iterations=500)

# Generate plots
plot_learning_curves(history)
plot_curriculum_heatmap(history)
plot_action_distributions(teacher)
```

### 4. Compare with Baselines

```python
from train_teacher import train_teacher, train_baseline_random, train_baseline_fixed
from visualize import plot_comparison

# Train all strategies
history_teacher, _, _ = train_teacher(num_iterations=500, verbose=False)
history_random = train_baseline_random(num_iterations=500)
history_fixed = train_baseline_fixed(num_iterations=500)

# Compare
plot_comparison({
    'teacher': history_teacher,
    'random': history_random,
    'fixed': history_fixed
})
```

## Architecture

### Components

1. **interfaces.py**: Shared data structures (Task, StudentState, TeacherAction) and ABC interfaces
2. **mock_student.py**: Student agent with learning (improves with practice) and forgetting (Ebbinghaus curve)
3. **mock_task_generator.py**: Simple task generator with 5 topics × 3 difficulties
4. **teacher_agent.py**: UCB bandit algorithm for selecting curriculum actions
5. **train_teacher.py**: Main training loop connecting all components
6. **test_teacher.py**: Unit tests for all components
7. **visualize.py**: Plotting utilities for analysis

### Action Space

Teacher selects from **30 actions**:
- 5 topics: history, science, literature, geography, current_events
- 3 difficulties: easy, medium, hard
- 2 options: new material or review

### Student Model

- **Learning**: Skill improves with practice: `new_skill = old_skill + learning_rate * difficulty_factor * (1 - old_skill)`
- **Forgetting**: Retention decays over time: `retention = exp(-forgetting_rate * time_since_practice)`
- **Effective Skill**: `effective_skill = base_skill * retention`
- **Accuracy**: `accuracy = 0.25 + 0.75 * effective_skill` (25% is random guessing on 4-choice MCQ)

### Teacher Algorithm

**UCB (Upper Confidence Bound)**:
```
UCB(a) = estimated_reward(a) + exploration_bonus × sqrt(log(total_pulls) / pulls(a))
```

- Balances exploration (trying new actions) vs exploitation (using known-good actions)
- Exploration bonus controls adventurousness (higher = more exploration)

### Reward Function

```
reward = improvement + difficulty_bonus + review_bonus + review_penalty

where:
- improvement = accuracy_after - accuracy_before
- difficulty_bonus = easy:0.5, medium:1.0, hard:2.0
- review_bonus = 1.0 if review and improvement > 0
- review_penalty = -0.5 if review and accuracy > 0.9 (wasted review)
```

## Expected Behavior

### Early Iterations (0-100)
- Teacher explores all topics/difficulties
- Tries mostly easy tasks (build foundation)
- High exploration, low exploitation

### Mid Iterations (100-300)
- Starts increasing difficulty
- Discovers which topics student struggles with
- Begins strategic reviewing

### Late Iterations (300-500)
- Mostly medium/hard tasks (student is skilled)
- Reviews topics just before forgetting threshold
- High exploitation of known-good curriculum

### Emergent Behaviors
- Teacher gives harder tasks as student improves
- Teacher reviews topics ~30-50 iterations after practice (optimal timing)
- Teacher specializes in topics student finds difficult

## Success Criteria

After training, you should see:
- ✅ Student reaches >70% accuracy by iteration 500
- ✅ Teacher discovers: easy tasks first → harder tasks later
- ✅ Teacher learns to review before forgetting
- ✅ Teacher reward stabilizes (not just random)

## File Structure

```
teacher_agent_dev/
├── interfaces.py           # Shared data structures and ABC interfaces
├── mock_student.py         # Mock student with learning + forgetting
├── mock_task_generator.py  # Simple task generator
├── teacher_agent.py        # MAIN: UCB bandit teacher algorithm
├── train_teacher.py        # Training loop
├── test_teacher.py         # Unit tests
├── visualize.py            # Plotting utilities
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Customization

### Adjust Student Learning
```python
student = MockStudentAgent(
    learning_rate=0.15,    # How fast student learns (higher = faster)
    forgetting_rate=0.05   # How fast student forgets (higher = faster)
)
```

### Adjust Teacher Exploration
```python
teacher = TeacherAgent(
    exploration_bonus=2.0  # Higher = more exploration, Lower = more exploitation
)
```

### Add More Topics/Difficulties
Edit `mock_task_generator.py` to add more templates or modify `teacher_agent.py` to adjust action space.

## Troubleshooting

**Issue**: Student doesn't learn
- **Solution**: Increase `learning_rate` in MockStudentAgent

**Issue**: Teacher doesn't explore
- **Solution**: Increase `exploration_bonus` in TeacherAgent

**Issue**: Forgetting too fast/slow
- **Solution**: Adjust `forgetting_rate` in MockStudentAgent

**Issue**: Division by zero errors
- **Solution**: UCB handles cold start automatically (untried actions selected first)

## Next Steps

1. **Replace mock components**: When teammates finish real student/task generator, swap out mock components
2. **Tune hyperparameters**: Adjust learning_rate, forgetting_rate, exploration_bonus
3. **Experiment with algorithms**: Try different bandit algorithms (Thompson Sampling, ε-greedy)
4. **Add features**: More sophisticated reward functions, state representations, etc.

## License

MIT

