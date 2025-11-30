# Student Language Model Agent

DistilBERT-based student agent with online learning and memory decay for AI teacher-student system.

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run tests:

```bash
python test_student.py
```

3. Train student:

```bash
python train_student.py
```

4. Check visualizations:

```bash
ls student_visualizations/
```

## Features

- **Online Learning**: Fine-tunes on 1 task at a time (not batches)
- **Memory Decay**: Realistic forgetting using Ebbinghaus curves
- **Per-Topic Tracking**: Monitors progress separately for each topic
- **Comprehensive Metrics**: Learning rate, sample efficiency, retention analysis
- **Beautiful Visualizations**: 6+ publication-quality plots

## Integration with Other Components

### With Real Teacher Agent:

Replace `MockTeacherAgent` with real `TeacherAgent` in `train_student.py`

### With Real Task Generator:

Replace `MockTaskGenerator` with real `TaskGenerator` in `train_student.py`

### Interface Compatibility:

All components follow the interfaces in `interfaces.py` - as long as the interface is respected, components are plug-and-play.

## Key Parameters

- `learning_rate`: How fast student learns (default: 5e-5)
- `retention_constant`: Forgetting speed (default: 80.0, higher = slower forgetting)
- `max_length`: Max tokens for passage+question (default: 256)
- `gradient_accumulation_steps`: Stability for online learning (default: 4)

## Metrics Generated

- Overall accuracy curve
- Per-topic learning curves
- Retention/forgetting analysis
- Difficulty progression
- Topic distribution
- Sample efficiency (tasks to reach milestones)

## File Structure

- `student_agent.py` - Main DistilBERT student
- `memory_decay.py` - Ebbinghaus forgetting model
- `student_metrics.py` - Metrics tracking
- `visualize_student.py` - Plotting utilities
- `train_student.py` - Training script
- `test_student.py` - Unit tests
- `mock_teacher.py` - Dummy teacher for testing
- `mock_task_generator.py` - Dummy task generator for testing

## Expected Behavior

Student should:

1. Start at ~25% accuracy (random guessing on 4-choice MCQ)
2. Improve to 70-80% with practice
3. Forget over time when topics not reviewed
4. Learn faster on easy tasks, slower on hard tasks
5. Show per-topic specialization

## Troubleshooting

**Student not improving:**
- Increase `learning_rate` (try 1e-4)
- Train for more iterations
- Check task quality

**Forgetting too fast/slow:**
- Adjust `retention_constant`
- Higher value = slower forgetting

**Out of memory:**
- Use `device='cpu'`
- Reduce `max_length`
- Increase `gradient_accumulation_steps`

