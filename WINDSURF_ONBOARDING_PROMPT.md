# MentorFlow - Complete Project Summary for Windsurf IDE

## 🎯 Project Overview

**MentorFlow** (formerly TeachRL) is a **meta-curriculum reinforcement learning system** where an **AI Teacher Agent** (using multi-armed bandit algorithms) learns to select optimal educational tasks to train an **AI Student Agent**, achieving faster learning than random or fixed curricula.

### Core Concept

The system implements a meta-learning loop:
1. **Teacher** selects a task (topic + difficulty + review option)
2. **Student** learns on that task
3. **Teacher** receives reward based on student improvement
4. **Teacher** updates its policy and learns to sequence curriculum optimally
5. Repeat

This creates a **teacher that learns to teach** - optimizing curriculum over time based on student performance feedback.

---

## 🏗️ System Architecture

### Three Main Components

1. **Task Generator** (`tasks/task_generator.py`)
   - Generates coding reasoning tasks across 5 families × 3 difficulties = 15 task types
   - Ensures no answer leakage, deterministic generation
   - Task families: `var_trace`, `if_cond`, `loop_count`, `list_index`, `bool_logic`

2. **Student Agent** (`student/`)
   - **Production version**: PPO agent using Stable-Baselines3 (`ppo_agent.py`, `student_env.py`)
   - **Development version**: DistilBERT-based language model (`student_agent_dev/`)
   - Learns to solve coding tasks through practice

3. **Teacher Agent** (`teacher/` and `teacher_agent_dev/`)
   - **Production version**: UCB1/Thompson Sampling bandit (`teacher_bandit.py`)
   - **Development version**: Standalone UCB bandit with mock components (`teacher_agent_dev/`)
   - Selects optimal curriculum sequence based on student feedback

### Two Development Tracks

The project has **two parallel development tracks** for modular team work:

#### Track 1: **Teacher Agent Development** (`teacher_agent_dev/`)
- **Status**: ✅ Complete and working
- **Purpose**: Independent teacher agent system with mock student/task generator
- **Algorithm**: UCB (Upper Confidence Bound) multi-armed bandit
- **Action Space**: 15 topics × 7 difficulties × 2 options = 210 actions
- **Key Files**:
  - `teacher_agent.py` - UCB bandit implementation
  - `mock_student.py` - Student with learning + forgetting (Ebbinghaus curve)
  - `mock_task_generator.py` - Task generator (15 topics, 7 difficulty levels)
  - `train_teacher.py` - Training loop
  - `compare_strategies.py` - Compare Teacher vs Random vs Progressive strategies
  - `visualize.py` - Plotting utilities

#### Track 2: **Student Agent Development** (`student_agent_dev/`)
- **Status**: ✅ Complete and working
- **Purpose**: DistilBERT-based student with online learning and memory decay
- **Features**: 
  - Online fine-tuning (1 task at a time)
  - Ebbinghaus forgetting curve
  - Per-topic skill tracking
  - Comprehensive metrics and visualizations
- **Key Files**:
  - `student_agent.py` - DistilBERT model with online learning
  - `memory_decay.py` - Ebbinghaus forgetting implementation
  - `student_metrics.py` - Metrics tracking
  - `train_student.py` - Training script
  - `visualize_student.py` - 6+ publication-quality plots
  - `test_student.py` - Unit tests (with tqdm progress bars)

---

## 📁 Complete File Structure

```
MentorFlow/
│
├── 📄 README.md                          # Main project documentation
├── 📄 requirements.txt                   # Python dependencies
│
├── 🎯 Core Production Components
│   ├── tasks/
│   │   ├── task_generator.py            # 15 task types (5 families × 3 difficulties)
│   │   └── encoding.py                  # Task encoding utilities
│   │
│   ├── student/
│   │   ├── student_env.py               # Gymnasium environment for student
│   │   └── ppo_agent.py                 # PPO agent wrapper (Stable-Baselines3)
│   │
│   ├── teacher/
│   │   ├── teacher_bandit.py            # UCB1, Thompson Sampling, ε-greedy
│   │   ├── teacher_ucb.py               # UCB implementation
│   │   └── meta_curriculum.py           # Meta-curriculum logic
│   │
│   ├── training/
│   │   ├── training_loop.py             # Meta-training loop
│   │   ├── train_single_task.py         # Train PPO on single task
│   │   ├── train_with_eval_logging.py   # Training with periodic evaluation
│   │   ├── train_with_teacher.py        # Teacher-guided training
│   │   ├── evaluate.py                  # Evaluation utilities
│   │   ├── callbacks.py                 # Progress callbacks (RolloutProgressCallback)
│   │   └── run_experiments.py           # CLI experiment runner
│   │
│   ├── evaluation/
│   │   ├── eval_metrics.py              # Accuracy, AUC, learning speed
│   │   └── baselines.py                 # Random, majority baselines
│   │
│   └── viz/
│       ├── plot_learning.py             # Learning curves, dashboards
│       └── heatmap.py                   # Curriculum heatmaps
│
├── 🔬 Teacher Agent Development (Track 1)
│   └── teacher_agent_dev/
│       ├── interfaces.py                # Shared data structures (Task, StudentState, TeacherAction)
│       ├── teacher_agent.py             # ⭐ MAIN: UCB bandit RL algorithm
│       ├── mock_student.py              # Student with learning + forgetting + PPO-like features
│       ├── mock_task_generator.py       # 15 topics × 7 difficulties (210 actions)
│       ├── train_teacher.py             # Training loop with baselines
│       ├── compare_strategies.py        # Compare Teacher vs Random vs Progressive
│       ├── test_teacher.py              # Unit tests (all passing)
│       ├── visualize.py                 # Plotting utilities
│       ├── requirements.txt             # Dependencies
│       ├── README.md                    # Teacher agent documentation
│       └── comparison_all_strategies.png # Generated comparison plot
│
├── 🎓 Student Agent Development (Track 2)
│   └── student_agent_dev/
│       ├── interfaces.py                # Shared interfaces
│       ├── student_agent.py             # ⭐ MAIN: DistilBERT with online learning
│       ├── memory_decay.py              # Ebbinghaus forgetting curve
│       ├── student_metrics.py           # Comprehensive metrics tracking
│       ├── mock_teacher.py              # Dummy teacher for testing
│       ├── mock_task_generator.py       # Dummy task generator
│       ├── train_student.py             # Training script
│       ├── visualize_student.py         # 6+ visualization plots
│       ├── test_student.py              # Unit tests with tqdm progress bars
│       ├── requirements.txt             # Dependencies (includes tqdm>=4.65.0)
│       ├── README.md                    # Student agent documentation
│       ├── PERFORMANCE_NOTES.md         # Why tests are slow
│       └── TEST_OPTIMIZATION.md         # Test optimization summary
│
├── 🧪 Testing
│   └── tests/
│       ├── test_system.py               # System-wide tests
│       ├── test_task_generator.py       # Task generator tests
│       └── verify_components.py         # Component verification
│
├── 🚀 Scripts & Utilities
│   └── scripts/
│       └── diagnose_training.py         # Training diagnostics
│
├── 📊 Models & Results
│   └── models/
│       ├── ppo_*.zip                    # Trained PPO models
│       ├── *_summary.txt                # Training summaries
│       └── *.png                        # Generated plots
│
├── 🎨 Frontend/Demo
│   ├── demo/
│   │   └── app.py                       # Flask demo application
│   └── frontend/                        # Next.js frontend
│
└── 📝 Documentation
    ├── CHANGES_SUMMARY.md               # Recent changes summary
    ├── WINDSURF_ONBOARDING_PROMPT.md    # This file!
    └── [various READMEs in subdirectories]
```

---

## 🔑 Key Technical Details

### 1. Teacher Agent (teacher_agent_dev/)

**Algorithm**: Upper Confidence Bound (UCB) Multi-Armed Bandit

**Action Space**: 
- **210 actions** = 15 topics × 7 difficulties × 2 options (new/review)
- Topics: history, science, literature, geography, current_events, mathematics, programming, philosophy, art, music, biology, chemistry, physics, economics, psychology
- Difficulties: trivial, easy, medium, hard, expert, master, grandmaster

**UCB Formula**:
```
UCB(a) = estimated_reward(a) + exploration_bonus × sqrt(log(total_pulls) / pulls(a))
```

**Reward Function**:
```python
reward = improvement + difficulty_bonus + review_bonus + review_penalty
where:
- improvement = accuracy_after - accuracy_before
- difficulty_bonus = {trivial:0.2, easy:0.5, medium:1.0, hard:2.0, expert:3.0, master:4.0, grandmaster:5.0}
- review_bonus = 1.0 if review and improvement > 0
- review_penalty = -0.5 if review and accuracy > 0.9
```

**Key Features**:
- ✅ Verified RL: Teacher learns and improves over time
- ✅ Transfer learning in mock student (skills transfer between related topics)
- ✅ Exponential learning under coherent curriculum
- ✅ Curriculum coherence detection

### 2. Student Agent (student_agent_dev/)

**Model**: DistilBERT (HuggingFace Transformers)

**Architecture**:
- DistilBertForMultipleChoice for MCQ tasks
- Online learning (fine-tune on 1 task at a time)
- Gradient accumulation for stability

**Memory Decay** (Ebbinghaus Forgetting Curve):
```python
retention = exp(-time_since_practice / retention_constant)
effective_skill = base_skill * retention
```

**Key Parameters**:
- `learning_rate`: 5e-5 (default)
- `retention_constant`: 80.0 (higher = slower forgetting)
- `max_length`: 256 tokens
- `gradient_accumulation_steps`: 4

### 3. Production Student (student/)

**Implementation**: Stable-Baselines3 PPO

**Environment**: Custom Gymnasium environment (`StudentEnv`)
- Multi-step episodes with gradual observation reveal
- Reward only on final step (prevents answer leakage)
- Observation masking for progressive difficulty

**Training**:
- Progress callbacks: `RolloutProgressCallback` for accurate timestep tracking
- Shared progress for nested loops: `SharedProgressCallback`

### 4. Task Generator (tasks/task_generator.py)

**Task Structure**:
- 5 families: `var_trace`, `if_cond`, `loop_count`, `list_index`, `bool_logic`
- 3 difficulties: 0 (easy), 1 (medium), 2 (hard)
- 15 total task types

**Features**:
- Deterministic generation (seed-based)
- No answer leakage in observation vectors
- Procedural generation (not templates)

---

## 🚀 How to Run Things

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# For teacher agent development
cd teacher_agent_dev
pip install -r requirements.txt
python test_teacher.py                    # Run tests
python train_teacher.py                   # Train teacher agent
python compare_strategies.py              # Compare strategies

# For student agent development
cd student_agent_dev
pip install -r requirements.txt
python test_student.py                    # Run tests (with progress bars!)
python train_student.py                   # Train student agent

# Production training
python training/train_single_task.py      # Train on single task
python training/train_with_teacher.py     # Teacher-guided training
```

### Key Commands

**Teacher Agent Comparison**:
```bash
cd teacher_agent_dev
python compare_strategies.py --iterations 500 --deterministic
# Options:
#   --seed N        # Use specific seed
#   --iterations N  # Number of iterations
#   --deterministic # Use fixed seed=42
#   --runs N        # Multiple runs for variance
```

**Student Agent Training**:
```bash
cd student_agent_dev
python train_student.py
# Generates visualizations in student_visualizations/

# Run tests with progress bars
python test_student.py
```

**Production Training**:
```bash
# Train single task
python training/train_single_task.py --family 0 --difficulty 0 --total_timesteps 50000

# Train with teacher
python training/train_with_teacher.py
```

---

## ⚙️ Important Configuration

### Random Strategy (compare_strategies.py)

The random strategy **already correctly** picks:
- Random topic from all available topics
- Random difficulty from all available difficulties
- Independently (no correlation)

See lines 93-95 in `teacher_agent_dev/compare_strategies.py`:
```python
# Random strategy: choose random topic AND random difficulty independently
topic = rng.choice(topics)           # Random topic
difficulty = rng.choice(difficulties)  # Random difficulty
```

### Test Performance

**Why tests are slow**:
- DistilBERT model loading: 10-30 seconds (first time), 5-10s (cached)
- Not your laptop's fault - neural networks are just large!
- Each test creates new StudentAgent (loads model)

**Solutions implemented**:
- ✅ tqdm progress bars added to `test_student.py`
- ✅ Reduced iteration counts
- ✅ Better progress messages
- ✅ Graceful fallback if model fails to load

**Expected times**:
- Model loading: 10-30s (first), 5-10s (cached)
- Per test: 5-15s (with model)
- Total suite: 30-90s (with model)
- Without model (dummy): <5s

---

## 🔄 Development Workflow

### Team Organization

The project is organized for **modular team development**:

1. **Task Generator Team**: Works on `tasks/task_generator.py`
2. **Student Agent Team**: Works on `student_agent_dev/` (can swap to real `student/` later)
3. **Teacher Agent Team**: Works on `teacher_agent_dev/` (your responsibility!)

### Integration Points

**All components follow interfaces** defined in:
- `teacher_agent_dev/interfaces.py` (teacher track)
- `student_agent_dev/interfaces.py` (student track)

**To integrate**:
- Replace mock components with real ones
- Ensure interfaces match (Task, StudentState, TeacherAction)
- Components are plug-and-play as long as interfaces are respected

### Recent Changes

1. ✅ **Random strategy**: Already correctly implements random topic + random difficulty
2. ✅ **Test optimization**: Added tqdm progress bars to `test_student.py`
3. ✅ **Teacher expansion**: Expanded to 15 topics × 7 difficulties
4. ✅ **Student enhancement**: Added PPO-like features (transfer learning, exponential learning)
5. ✅ **Comparison plots**: Fixed accuracy drops, added curriculum diversity plot

---

## 📊 Key Metrics & Visualizations

### Teacher Agent (`teacher_agent_dev/`)

**Comparison Plot** (`comparison_all_strategies.png`):
- General accuracy over time
- Difficult question accuracy
- Curriculum diversity (unique topics explored)
- Learning efficiency

**Strategies compared**:
1. **Random**: Random topic + random difficulty
2. **Progressive**: Easy → Medium → Hard sequentially
3. **Teacher**: RL agent learns optimal curriculum (should perform best!)

### Student Agent (`student_agent_dev/`)

**Generated Visualizations** (`student_visualizations/`):
- Learning curve (overall accuracy)
- Per-topic learning curves
- Retention/forgetting analysis
- Difficulty progression
- Topic distribution
- Sample efficiency

---

## 🐛 Known Issues & Notes

### Performance

1. **DistilBERT loading is slow**: This is normal, not a bug
   - First run downloads ~260MB model
   - Subsequent runs use cache (faster)
   - Progress bars added to show what's happening

2. **Test slowness**: Due to model loading, not code inefficiency
   - Tests create new StudentAgent each time
   - Consider sharing instances in future optimization

### Accuracy Drops

**Fixed in compare_strategies.py**:
- Issue: Accuracy dropped at end of iterations
- Cause: Forgetting rate too aggressive, dynamic eval sets
- Solution: Reduced forgetting rate (0.05 → 0.01), fixed eval sets

### Progress Bars

**RolloutProgressCallback** (`training/callbacks.py`):
- Fixes "frozen progress bar" issue
- Tracks timesteps correctly using rollout events
- Use `RolloutProgressCallback` for SB3 training
- Use `SharedProgressCallback` for nested loops

---

## 🎯 Success Criteria

### Teacher Agent

After training teacher agent, you should see:
- ✅ Student reaches >70% accuracy by iteration 500
- ✅ Teacher discovers: easy tasks first → harder tasks later
- ✅ Teacher learns to review before forgetting
- ✅ Teacher reward stabilizes (not just random)
- ✅ Teacher outperforms random and progressive strategies

### Student Agent

After training student agent, you should see:
- ✅ Starts at ~25% accuracy (random guessing)
- ✅ Improves to 70-80% with practice
- ✅ Forgets over time when topics not reviewed
- ✅ Learns faster on easy tasks, slower on hard tasks
- ✅ Shows per-topic specialization

---

## 📦 Dependencies

### Main Project
- `stable-baselines3` - PPO implementation
- `gymnasium` - Environment interface
- `numpy`, `matplotlib`, `seaborn` - Data & visualization

### Teacher Agent Dev
- `numpy` - Numerical operations
- `matplotlib`, `seaborn` - Plotting

### Student Agent Dev
- `torch>=2.0.0` - PyTorch
- `transformers>=4.30.0` - HuggingFace (DistilBERT)
- `tqdm>=4.65.0` - Progress bars
- `numpy`, `matplotlib`, `seaborn` - Data & visualization

---

## 🔗 Integration Roadmap

### Current State

✅ **Teacher Agent**: Complete and working with mock components
✅ **Student Agent**: Complete and working with mock components
✅ **Task Generator**: Production version ready

### Next Steps

1. **Replace Mock Student**: Swap `mock_student.py` with real `student/ppo_agent.py`
   - Ensure `StudentAgentInterface` is implemented
   - Test with teacher agent

2. **Replace Mock Task Generator**: Swap `mock_task_generator.py` with real `tasks/task_generator.py`
   - Ensure `TaskGeneratorInterface` is implemented
   - Map task families to topics

3. **Unified System**: Integrate all three components
   - Teacher agent selects tasks
   - Real task generator creates tasks
   - Real student agent learns

4. **Production Training**: Use `training/train_with_teacher.py` with all real components

---

## 🎓 Key Concepts

### Meta-Learning
The teacher learns **how to teach** by optimizing curriculum selection based on student feedback.

### Curriculum Learning
Sequencing tasks from easy → hard, but **learned dynamically** rather than fixed.

### Multi-Armed Bandit
Teacher uses UCB bandit to balance:
- **Exploration**: Try new task types
- **Exploitation**: Use known-good task types

### Memory Decay (Ebbinghaus)
Students forget over time:
```
retention = exp(-time / retention_constant)
```
Teacher must review topics before students forget.

### Transfer Learning
Skills in related topics boost learning in new topics:
- STEM topics (math, science, programming) transfer to each other
- Humanities topics (history, literature) transfer to each other

---

## 🚨 Important Files to Know

### Must-Read Files

1. **`teacher_agent_dev/teacher_agent.py`** - Main teacher algorithm
2. **`teacher_agent_dev/interfaces.py`** - Component interfaces
3. **`teacher_agent_dev/compare_strategies.py`** - Strategy comparison
4. **`student_agent_dev/student_agent.py`** - Student implementation
5. **`tasks/task_generator.py`** - Task generation logic

### Configuration Files

- `requirements.txt` - Main dependencies
- `teacher_agent_dev/requirements.txt` - Teacher dev dependencies
- `student_agent_dev/requirements.txt` - Student dev dependencies

### Documentation

- `README.md` - Main project overview
- `teacher_agent_dev/README.md` - Teacher agent details
- `student_agent_dev/README.md` - Student agent details
- `CHANGES_SUMMARY.md` - Recent changes

---

## 💡 Tips for Windsurf

1. **Start with teacher_agent_dev/**: This is the most complete and working system
2. **Run tests first**: `python test_teacher.py` to verify everything works
3. **Check comparison plot**: `comparison_all_strategies.png` shows teacher outperforming baselines
4. **Use progress bars**: Tests show tqdm progress bars for slow operations
5. **Interface-based design**: Components are plug-and-play via interfaces
6. **Mock components**: Allow independent development (your track = teacher agent)

---

## 📝 Final Notes

- **Project Status**: ✅ Working and complete
- **Teacher Agent**: ✅ Verified RL, learns optimal curriculum
- **Student Agent**: ✅ DistilBERT with online learning and memory decay
- **Integration**: Ready to swap mock components for real ones
- **Performance**: Tests are slow due to model loading (normal, not a bug)

**You're ready to continue development!** The teacher agent system is complete and working. The next step is integrating with real student and task generator components when teammates finish their work.

---

## 🔍 Quick Reference

**Run teacher comparison**:
```bash
cd teacher_agent_dev && python compare_strategies.py
```

**Run student tests**:
```bash
cd student_agent_dev && python test_student.py
```

**Train teacher**:
```bash
cd teacher_agent_dev && python train_teacher.py
```

**Train student**:
```bash
cd student_agent_dev && python train_student.py
```

**View plots**:
- Teacher: `teacher_agent_dev/comparison_all_strategies.png`
- Student: `student_agent_dev/student_visualizations/`

---

*This document summarizes the entire MentorFlow project as of the migration to Windsurf IDE. All components are working and ready for continued development.*

