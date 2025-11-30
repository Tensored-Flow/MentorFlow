---
title: MentorFlow
emoji: 🎓
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
hardware: gpu-t4
---

# MentorFlow - Teacher-Student RL System

A meta-curriculum reinforcement learning system where an AI Teacher Agent learns to select optimal educational tasks to train an AI Student Agent.

## 🚀 Features

- **Three Training Strategies**: Compare Random, Progressive, and Teacher-guided curriculum
- **LM Student (DistilBERT)**: Real neural network learning with memory decay
- **GPU Support**: Fast training with CUDA acceleration
- **Interactive Comparison**: Visualize learning curves and performance metrics

## 📊 Usage

1. **Set Parameters**:
   - Iterations: Number of training iterations (50-500)
   - Seed: Random seed for reproducibility
   - Device: Choose GPU (cuda) or CPU

2. **Run Comparison**:
   - Click "Run Comparison" to start training
   - Monitor progress in the output text
   - View generated comparison plots

3. **Analyze Results**:
   - Learning curves show how each strategy improves
   - Difficult question performance shows final accuracy
   - Curriculum diversity shows topic coverage

## ⚡ Performance

- **With GPU**: ~5-10 minutes for 500 iterations
- **With CPU**: ~15-30 minutes for 500 iterations

## 📁 Project Structure

```
MentorFlow/
├── app.py                      # Gradio web interface
├── teacher_agent_dev/          # Teacher agent system
│   ├── compare_strategies.py  # Main comparison script
│   ├── teacher_agent.py       # UCB bandit teacher
│   └── ...
├── student_agent_dev/          # LM Student system
│   ├── student_agent.py       # DistilBERT student
│   └── ...
└── requirements_hf.txt        # Dependencies
```

## 🔧 Technical Details

- **Teacher Agent**: UCB (Upper Confidence Bound) multi-armed bandit
- **Student Agent**: DistilBERT with online learning
- **Memory Decay**: Ebbinghaus forgetting curve
- **Task Generator**: Procedural generation with 15 topics × 7 difficulties

## 📖 More Information

See the main repository for detailed documentation and development guides.

