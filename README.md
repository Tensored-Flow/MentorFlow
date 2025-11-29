# TeachRL – AI Teacher That Learns to Teach Coding

A meta-curriculum reinforcement learning system where an **AI Teacher** (multi-armed bandit) learns to select optimal coding microtasks to train an **AI Student** (PPO), achieving faster learning than random or fixed curricula.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    META-TRAINING LOOP                       │
├─────────────────────────────────────────────────────────────┤
│  1. Evaluate Student → acc_before                           │
│  2. Teacher selects arm (task type) → arm_id                │
│  3. Student trains on task for K steps                      │
│  4. Evaluate Student → acc_after                            │
│  5. Teacher reward = acc_after - acc_before                 │
│  6. Repeat                                                  │
└─────────────────────────────────────────────────────────────┘
```

### Components

| Component | Description |
|-----------|-------------|
| **Task Generator** | 5 families × 3 difficulties = 15 task types |
| **Student (PPO)** | Learns to solve coding tasks |
| **Teacher (Bandit)** | UCB1/Thompson Sampling to select curriculum |
| **Baselines** | Random, Fixed (easy→hard) curriculum |

### Task Families

| Family | Description |
|--------|-------------|
| `var_trace` | Track variable values through assignments |
| `if_cond` | Evaluate conditional expressions |
| `loop_count` | Count loop iterations |
| `list_index` | List indexing and access |
| `bool_logic` | Boolean expression evaluation |

## 📁 Project Structure

```
MentorFlow/
├── tasks/
│   └── task_generator.py    # 15 task types with procedural generation
├── student/
│   ├── student_env.py       # Gym environment for student
│   └── ppo_agent.py         # PPO agent wrapper (SB3)
├── teacher/
│   └── teacher_bandit.py    # UCB1, Thompson, ε-greedy bandits
├── training/
│   ├── training_loop.py     # Meta-training loop
│   └── run_experiments.py   # CLI experiment runner
├── evaluation/
│   ├── eval_metrics.py      # Accuracy, AUC, learning speed
│   └── baselines.py         # Random, majority baselines
├── viz/
│   ├── plot_learning.py     # Learning curves, dashboards
│   └── heatmap.py           # Curriculum heatmaps
└── requirements.txt
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run a Quick Test

```bash
# Test task generator
python tasks/task_generator.py

# Test student environment
python student/student_env.py

# Test teacher bandit
python teacher/teacher_bandit.py
```

### Run Meta-Training

```bash
# Single strategy (UCB1)
python training/run_experiments.py --mode single --strategy ucb1 --meta-steps 100

# Compare all strategies
python training/run_experiments.py --mode compare --meta-steps 200
```

### CLI Options

```bash
python training/run_experiments.py \
  --mode compare \           # 'single' or 'compare'
  --strategies ucb1 random fixed \
  --meta-steps 200 \         # Number of meta-training steps
  --train-steps 256 \        # PPO steps per meta-step
  --eval-tasks 10 \          # Eval tasks per type (150 total)
  --seed 42 \
  --output results/
```

## 📊 Example Output

```
============================================================
COMPARISON SUMMARY
============================================================
Strategy             Final Acc    Total Reward       Time
------------------------------------------------------------
ucb1                    68.00%          0.4300       45.2s
thompson_sampling       65.33%          0.4033       44.8s
fixed                   60.00%          0.3500       43.1s
random                  55.33%          0.3033       42.9s
```

## 🔬 Key APIs

### Generate a Task
```python
from tasks.task_generator import generate_task, generate_task_by_arm

task = generate_task(family_id=0, difficulty_id=1, seed=42)
# or
task = generate_task_by_arm(arm_id=7, seed=42)

print(task.human_prompt)
print(task.human_choices)
print(f"Correct: {task.correct_action}")
```

### Train Student on Task
```python
from student.ppo_agent import StudentAgent

student = StudentAgent(seed=42)
student.train_on_task(arm_id=3, total_timesteps=256)
accuracy = student.evaluate(eval_tasks)
```

### Teacher Selects Curriculum
```python
from teacher.teacher_bandit import TeacherBandit, BanditStrategy

teacher = TeacherBandit(strategy=BanditStrategy.UCB1)
arm_id = teacher.select_arm()
# ... student trains ...
teacher.update(arm_id, reward=0.02)  # improvement
```

### Run Full Meta-Training
```python
from training.training_loop import MetaTrainer, MetaTrainingConfig

config = MetaTrainingConfig(num_meta_steps=200, seed=42)
trainer = MetaTrainer(config)
results = trainer.train()

print(f"Final accuracy: {results.final_accuracy:.2%}")
```

## 📈 Visualizations

```python
from viz.plot_learning import plot_comparison, create_dashboard
from viz.heatmap import plot_curriculum_heatmap

# Learning curve comparison
plot_comparison(results_dict, save_path="learning_curves.png")

# Curriculum heatmap
plot_curriculum_heatmap(results.curriculum_heatmap, save_path="heatmap.png")

# Full dashboard
create_dashboard(results_dict, save_path="dashboard.png")
```

## 🎯 Hackathon Deliverables

- [x] Task Generator (5 families × 3 difficulties)
- [x] Student Environment (Gym-compatible)
- [x] PPO Student Agent (Stable-Baselines3)
- [x] Teacher Bandit (UCB1, Thompson, ε-greedy)
- [x] Meta-Training Loop
- [x] Baseline Comparisons
- [x] Evaluation Metrics
- [x] Visualization Tools

## 📝 License

MIT
