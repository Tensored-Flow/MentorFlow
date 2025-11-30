# Deploying MentorFlow to Hugging Face Spaces

## 🚀 Quick Deploy Guide

### Step 1: Clone Your Space

```bash
# Using git
git clone https://huggingface.co/spaces/iteratehack/MentorFlow
cd MentorFlow

# OR using HF CLI
hf download iteratehack/MentorFlow --repo-type=space
```

### Step 2: Copy Files to Space

Copy these files from your local MentorFlow repository to the HF Space:

**Required Files:**
```
MentorFlow/
├── app.py                      # ✅ Gradio interface (NEW)
├── requirements_hf.txt         # ✅ Dependencies (NEW)
├── README_HF_SPACE.md         # ✅ Space README (NEW)
├── teacher_agent_dev/          # ✅ Entire directory
│   ├── compare_strategies.py
│   ├── teacher_agent.py
│   ├── mock_task_generator.py
│   ├── mock_student.py
│   ├── interfaces.py
│   └── ... (all files)
└── student_agent_dev/          # ✅ Entire directory
    ├── student_agent.py
    ├── memory_decay.py
    ├── interfaces.py
    └── ... (all files)
```

**Optional (for better organization):**
- `README.md` - Project documentation
- `LICENSE` - License file

### Step 3: Update README.md

Replace the Space's README.md with content from `README_HF_SPACE.md`:

```bash
cp README_HF_SPACE.md README.md
```

The README.md should have this header:
```yaml
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
hardware: gpu-t4  # Request GPU!
---
```

### Step 4: Request GPU Access

In your Space settings on Hugging Face:
1. Go to Settings → Hardware
2. Select "GPU T4 Small" or "GPU T4 Medium"
3. Save settings

### Step 5: Commit and Push

```bash
git add .
git commit -m "Deploy MentorFlow comparison tool with GPU support"
git push
```

The Space will automatically build and deploy!

## 📝 File Structure in Space

```
MentorFlow/
├── app.py                      # Gradio interface
├── README.md                   # Space metadata + description
├── requirements.txt            # Rename from requirements_hf.txt
├── teacher_agent_dev/
│   └── ... (all files)
└── student_agent_dev/
    └── ... (all files)
```

## ⚙️ Configuration

### Hardware Selection

In README.md metadata:
- `hardware: gpu-t4` - Request GPU T4 Small (free tier)
- `hardware: gpu-t4-medium` - Request GPU T4 Medium (paid)

### Environment Variables

The app automatically detects GPU. To force CPU:
- Set `device="cpu"` in the Gradio interface

## 🐛 Troubleshooting

### Build Fails

1. Check `requirements.txt` syntax
2. Ensure all imports are available
3. Check logs in Space logs tab

### GPU Not Available

1. Check Space settings → Hardware
2. Verify `hardware: gpu-t4` in README.md
3. GPU may take time to provision

### Import Errors

1. Ensure all subdirectories are included
2. Check sys.path modifications in app.py
3. Verify relative imports work

### Timeout Issues

1. Reduce iterations for testing
2. Check Space timeout settings
3. Use progress updates to monitor

## 🎯 Testing Locally First

Before deploying, test locally:

```bash
# Install dependencies
pip install -r requirements_hf.txt

# Run Gradio app
python app.py

# Open browser to http://localhost:7860
```

## 📊 What Gets Deployed

✅ **Deployed:**
- Gradio web interface (`app.py`)
- Teacher agent system
- LM Student system  
- Comparison scripts
- All dependencies

❌ **Not Deployed:**
- Local models/trained weights
- Cache files (`__pycache__`)
- Virtual environment
- Test files (optional)

## 🔗 After Deployment

Your Space will be available at:
**https://huggingface.co/spaces/iteratehack/MentorFlow**

Share this link to let others run comparisons!

## 📝 Notes

- GPU is **recommended** for faster training (5-10 min vs 15-30 min)
- Free tier GPU may have usage limits
- Space automatically rebuilds on git push
- Logs available in Space logs tab

